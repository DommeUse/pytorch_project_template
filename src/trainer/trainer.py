import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        outputs = self.model(**batch)
        batch.update(outputs)

        if "perplexities" in outputs:
            mean_ppl = torch.stack(outputs["perplexities"]).mean().item()
            metrics.update("mean_perplexity", mean_ppl)

        if self.use_gan and self.is_train:
            real_features_d = self.discriminator(batch["audio"])
            fake_features_d = self.discriminator(batch["output"].detach())

            real_logits_d = [feats[-1] for feats in real_features_d]
            fake_logits_d = [feats[-1] for feats in fake_features_d]

            d_losses = self.disc_criterion(real_logits_d, fake_logits_d)

            self.optimizer_d.zero_grad()
            d_losses["loss"].backward()
            self.optimizer_d.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

            real_features = self.discriminator(batch["audio"])
            fake_features = self.discriminator(batch["output"])
            fake_logits = [feats[-1] for feats in fake_features]

            g_losses = self.criterion(
                audio = batch["audio"],
                output = batch["output"],
                commitment_loss = batch["commitment_loss"],
                fake_logits = fake_logits,
                real_features = real_features,
                fake_features = fake_features,
            )

            self.optimizer.zero_grad()
            g_losses["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            batch.update(g_losses)
            batch["d_loss"] = d_losses["loss"]

        elif self.is_train:
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            self.optimizer.zero_grad()
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        else:
            rec = self.criterion.rec_loss(batch["output"], batch["audio"])
            batch["g_rec"] = rec
            batch["g_commit"] = batch["commitment_loss"]
            batch["loss"] = rec

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        if self.writer == None:
            return

        target = batch["audio"][0].detach().cpu()
        result = batch["output"][0].detach().cpu()

        self.writer.add_audio(f"{mode}/target", target, 16000)
        self.writer.add_audio(f"{mode}/reconstructed", result, 16000)

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
