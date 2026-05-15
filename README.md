# SoundStream - нейросетевой аудио-кодек

Реализация [SoundStream (Zeghidour et al., 2021)](https://arxiv.org/abs/2107.03312) для домашнего задания №4 по курсу Глубинного обучения.

Проект построен на основе [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) с использованием [Hydra](https://hydra.cc/docs/intro/) для конфигурации и [Comet ML](https://www.comet.com/) для логирования.

## Содержание

- [Архитектура](#архитектура)
- [Структура репозитория](#структура-репозитория)
- [Установка](#установка)
- [Тренировка](#тренировка)
- [Инференс](#инференс)
- [Результаты](#результаты)
- [Comet эксперименты](#comet-эксперименты)

## Архитектура

| Компонент | Параметры |
|---|---|
| Encoder | каузальный SEANet, страйды `[2, 4, 5, 5]`, base channels = 32 |
| Residual unit | kernel_size 7, dilation `(1, 3, 9)`, pre-activation, ELU |
| RVQ | 8 квантизаторов $\times$ размер кодбука 1024, EMA decay 0.99, k-means инициализация, dead-code-reset с порогом  2 |
| Decoder | зеркальный энкодеру, страйды `[5, 5, 4, 2]`, `CausalConvTranspose1d` |
| Latent dim | 128 |
| Wave discriminator | multi-scale (3 разрешения), 1D-свёртки, LeakyReLU 0.2, weight norm |
| STFT discriminator | n_fft = 1024, hop_length = 256, 2-канальный complex spectrum (real+imag), drop DC bin |
| Reconstruction loss | multi-scale mel-spectrogram, $s \in \{2^6, 2^7, 2^8, 2^9, 2^{10}, 2^{11}\}$, 64 mel-бина |
| Adversarial loss | hinge GAN на всех дискриминаторах |
| Feature matching | mean L1-loss между `real_feat.detach()` и `fake_feat` |
| Веса лоссов | $\lambda_{adv} = 1, \lambda_{feat} = 100, \lambda_{rec} = 1, \lambda_{commit} = 1$ |
| Оптимизатор | Adam, lr = 1e-4(const), $\beta$ = (0.5, 0.9) |

Параметры тренировки: кропы по 0.5 секунд, batch size = 12, 45000 шагов на NVIDIA T4 (Kaggle) и NVIDIA A100(Colab).

## Структура репозитория

```
.
├── src/
│   ├── configs/                      # Hydra конфиги
│   │   ├── soundstream.yaml          # baseline без дискриминатора
│   │   ├── soundstream_gan.yaml      # полная GAN-тренировка
│   │   ├── onebatchtest.yaml         # sanity-check на 32 файлах
│   │   ├── onebatchtest_gan.yaml     # GAN sanity-check на 32 файлах
│   │   ├── inference.yaml            # конфиг для инференса
│   │   ├── model/                    # архитектура SoundStream
│   │   ├── discriminator/            # параметры MultiScaleDiscriminator
│   │   ├── loss_function/            # параметры GeneratorLoss, ReconstructionOnlyLoss
│   │   ├── disc_loss_function/       # параметры DiscriminatorLoss
│   │   ├── datasets/, dataloader/    # LibriSpeech
│   │   ├── transforms/               # обрезка и подготовка аудио
│   │   ├── metrics/                  # STOI, NISQA, MeanPerplexity
│   │   ├── optimizer_d/              # Adam для дискриминатора
│   │   └── writer/                   # Comet ML
│   ├── model/
│   │   ├── soundstream.py            # SoundStream (encoder + RVQ + decoder)
│   │   ├── encoder.py, decoder.py    # SEANet архитектура
│   │   ├── rvq.py                    # ResidualVQ
│   │   ├── blocks.py                 # все структурные блоки(ResidualUnit, CausalConv1d, и т.д.)
│   │   └── discriminators.py         # WaveDisc, STFTDisc, MultiScaleDisc
│   ├── loss/
│   │   ├── reconstruction.py         # MultiScaleMelLoss
│   │   ├── adversarial.py            # hinge / feature-matching функции
│   │   ├── generator.py              # GeneratorLoss
│   │   ├── discriminator.py          # DiscriminatorLoss
│   │   └── reconstruction_only.py    # ReconstructionOnlyLoss (для baseline без дискриминатора)
│   ├── metrics/
│   │   ├── stoi.py, nisqa_v2.py      # реализация метрик с помощью torchmetrics
│   │   └── perplexity.py             # утилизация кодбуков
│   ├── datasets/                     # LibriSpeech, collate_fn
│   ├── transforms/                   # AudioCutter
│   ├── trainer/
│   │   ├── base_trainer.py           # с поддержкой двух оптимизаторов
│   │   ├── trainer.py                # GAN-цикл + recon-only
│   │   └── inferencer.py             # для инференса
│   └── ...
├── scripts/                          # утилиты и sanity-тесты
├── train.py                          # точка входа для тренировки
├── inference.py                      # точка входа для инференса
├── requirements.txt
├── README.md                         # этот файл
└── REPORT.md                         # отчёт по работе и сложностям
```

## Установка

```bash
git clone -b soundstream-gan https://github.com/DommeUse/pytorch_project_template.git
cd pytorch_project_template
git checkout soundstream-gan

# создать виртуальное окружение (опционально)
python3 -m venv venv && source venv/bin/activate

pip install -r requirements.txt
```

## Тренировка

### Подготовка данных

LibriSpeech скачивается автоматически при первом запуске. Можно также указать путь к уже скачанным данным, добавив в `python train.py` следующие hydra-аргументы:

```bash
+datasets.train.data_dir="<LIBRISPEECH_PATH>" \
+datasets.test.data_dir="<LIBRISPEECH_PATH>"
# структура: <LIBRISPEECH_PATH>/{train-clean-100, test-clean}
```

### Comet ML

API-ключ лучше положить в переменную окружения:
```bash
export COMET_API_KEY=<your_key>
```

### Sanity check (One batch test)

Проверка, что pipeline собирается и обучение работает корректно:

```bash
# recon-only sanity (~500 шагов)
PYTHONPATH=. python train.py --config-name=onebatchtest writer.run_name="<exp-name>"

# GAN sanity (~1000 шагов)
PYTHONPATH=. python train.py --config-name=onebatchtest_gan writer.run_name="<exp-name>"
```

### Полная тренировка

**Reconstruction-only baseline** (без GAN):
```bash
PYTHONPATH=. python train.py \
    --config-name=soundstream \
    writer.run_name="<exp-name>"
```

**Полная GAN-тренировка**:
```bash
PYTHONPATH=. python train.py \
    --config-name=soundstream_gan \
    writer.run_name="<exp-name>"
```

### Kaggle

Предварительно в Kaggle Input добавьте датасет Librispeech - это освободит вас от необходимости его устанавливать при первом запуске.

Для запуска на Kaggle (T4):

```python
# В первой ячейке ноутбука:
!git clone -b soundstream-gan https://github.com/DommeUse/pytorch_project_template.git
%cd pytorch_project_template
!pip install -r requirements.txt

# Далее добавляем ваш ключ в переменную окружения

# 1 вариант
import os
os.environ["COMET_API_KEY"] = "<your_key>"
# 2 вариант(нужно предварительно его добавить в kaggle secrets)
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["COMET_API_KEY"] = user_secrets.get_secret("COMET_API_KEY")

# Запуск обучения

!python train.py \
    --config-name=soundstream_gan \
    trainer.save_dir="/kaggle/working/saved" \
    +datasets.train.data_dir="/kaggle/input/datasets/a24998667/librispeech" \
    +datasets.test.data_dir="/kaggle/input/datasets/a24998667/librispeech" \
    writer.log_checkpoints=True \ # для сохранения чекпоинтов
    writer.run_name="<exp-name>"
```

### Colab

В Google Colab нет встроенной базы датасетов, поэтому придется загружать датасет самому. В остальном шаги аналогичные.

Для запуска на Colab (T4):

```python
# В первой ячейке ноутбука:
!git clone -b soundstream-gan https://github.com/DommeUse/pytorch_project_template.git
%cd pytorch_project_template
!pip install -r requirements.txt

# Далее добавляем ваш ключ в переменную окружения

# 1 вариант
import os
os.environ["COMET_API_KEY"] = "<your_key>"
# 2 вариант(нужно предварительно его добавить в colab secrets)
import os
from google.colab import userdata

os.environ["COMET_API_KEY"] = userdata.get('COMET_API_KEY')

# Запуск обучения

!python train.py \
    --config-name=soundstream_gan \
    trainer.save_dir="/kaggle/working/saved" \
    writer.log_checkpoints=True \ # для сохранения чекпоинтов
    writer.run_name="<exp-name>"
```

## Инференс

### Загрузка модели

Преобученный чекпоинт модели загружен на Google Drive, поэтому сначала нам нужно его скачать с помощью библиотеки gdown

```bash
!pip install -q gdown
```

Далее выполняем код

```python
import gdown

CHECKPOINT_GDRIVE_ID = '1bVDPp12NwIwe9VNfy1DjGNi-Xk0HODYU'
CHECKPOINT_PATH = 'checkpoint.pth'

gdown.download(id = CHECKPOINT_GDRIVE_ID, output = CHECKPOINT_PATH, quiet = False)
```

Теперь мы можем загрузить веса самой модели

```python
import torch
from pathlib import Path
from src.model import SoundStream
from src.trainer import FileInferencer

# такой же сетап, как у преобученной модели
model = SoundStream(
    encoder_channels = 32,
    target_channels = 128,
    n_quantizers = 8,
    codebook_size = 1024,
).to(device)            

checkpoint = torch.load(CHECKPOINT_PATH, map_location = device, weights_only = False)
state_dict = checkpoint.get('state_dict', checkpoint)
model.load_state_dict(state_dict, strict = True) # загружаем веса
```

### Реконструкция аудио-файлов через обученную модель

Для этого есть класс inferencer.FileInferencer, с помощью которого можно реконструировать аудио-файлы.

```python
from inferencer import FileInferencer

save_path = Path("<путь к папке, куда вы хотите сохранить результаты>")
sample_rate = 16_000 # обученная модель работает с такой частотой

file_inferencer = FileInferencer(model, sample_rate, device, save_path)
```

```python
input_path = <путь к вашему файлу>
output_name = <имя файла для сохранения в save_path>
reconstructed = file_inferencer(input_path, output_name)
```


## Результаты

### Финальные метрики (на test-clean)

| Метрика | Recon-only baseline | Полный GAN | Порог в задании |
|---|---|---|---|
| STOI | **0.818** | **0.777** | > 0.80 |
| NISQA MOS | **1.654** | **2.596** | > 2.25 |
| Mean perplexity (8 квантизаторов) | **402.77** | **425** | - |

Подробное обсуждение результатов и сравнение GAN vs no-GAN - в [REPORT.md](./REPORT.md).

## Comet эксперименты

- Recon-only baseline: https://www.comet.com/german-zverev/soundstream/kcxmna44b39ehi51rp43aqmo16koj11r
- GAN финальный: https://www.comet.com/german-zverev/soundstream/1fn2j76oxlff2oclv45qtavignuxpu65

## Ссылки

- Шаблон проекта: [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template) by [Petr Grinberg](https://github.com/Blinorot)
- Архитектура SoundStream: Zeghidour et al., 2021, [arXiv:2107.03312](https://arxiv.org/abs/2107.03312)
- Архитектура SEANet: M. Tagliasacchi, Y. Li, K. Misiunas, and D. Roble, 2020, [arXiv:2009.02095](https://arxiv.org/abs/2009.02095)