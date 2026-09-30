# GlassesDet

Glasses detection with a fine-tuned YOLO12m model. Given an image or a webcam frame, the model locates each person's face region and classifies it as `wearing_glasses` or `not_wearing_glasses`.

The project starts from pretrained YOLO12m weights (Ultralytics) and fine-tunes them on a small Kaggle dataset. It includes dataset preparation, exploratory data analysis, training, evaluation on a held-out test set, and a real-time webcam demo.

## Results

Evaluation of the best checkpoint on the held-out test set (15 original images):

| Class               | Precision | Recall | F1    | mAP@0.5 | mAP@0.5:0.95 |
|---------------------|----------:|-------:|------:|--------:|-------------:|
| not_wearing_glasses | 0.732     | 0.714  | 0.723 | 0.820   | 0.410        |
| wearing_glasses     | 0.874     | 0.865  | 0.869 | 0.827   | 0.436        |
| All                 | 0.803     | 0.790  | 0.796 | 0.823   | 0.423        |

Inference takes about 20 ms per image at 640x640 on an RTX 4090 Laptop GPU.

The validation and test sets are small (12 and 15 images), so these figures carry a fair amount of variance. The gap between mAP@0.5 and mAP@0.5:0.95 shows the model reliably finds the right region, but its bounding boxes are often loosely fitted.

## Project structure

```text
GlassesDet/
├── Notebook_GlassesDet.ipynb     EDA, training, evaluation and analysis
├── live_inference.py             Real-time webcam inference
├── utils/
│   └── resplit_and_augment.py    Dataset re-split and training-set augmentation
├── yolo12m.pt                    Pretrained YOLO12m weights
├── weights/yolo26n.pt            Pretrained YOLO26n weights
├── pyproject.toml, uv.lock       Dependencies
├── data/                         Dataset (not included)
└── runs/                         Training outputs (not included)
```

The dataset and training outputs are not included in the repository because of their size.

## Installation

Requires Python 3.10 or newer. Dependencies are managed with [uv](https://docs.astral.sh/uv/); on Windows and Linux, PyTorch is installed with CUDA 12.6 support.

```bash
git clone https://github.com/oddissblue/GlassesDet.git
cd GlassesDet
uv sync
```

## Dataset

The project uses the [Glasses Detection (YOLO format)](https://www.kaggle.com/datasets/mohamedchahed/glasses-detection-yolo-format) dataset from Kaggle. It contains 134 images with YOLO annotations: 59 `not_wearing_glasses` and 75 `wearing_glasses`.

After downloading, extract the dataset to `data/DST_GlassesDet/` with the following layout:

```text
DST_GlassesDet/
├── train/{images,labels}
├── val/{images,labels}
├── test/{images,labels}
└── yolo.yaml
```

`yolo.yaml`:

```yaml
path: <absolute path to data/DST_GlassesDet>
train: train/images
val: val/images
test: test/images
names:
  0: not_wearing_glasses
  1: wearing_glasses
```

Then prepare the splits:

```bash
uv run python utils/resplit_and_augment.py
```

This script pools all images and makes a stratified 80/10/10 split (seed 42). It then augments only the training set, using horizontal flips, color jitter, blur and noise, and affine transforms from Albumentations, which grows it from 107 to 321 images. The validation and test sets contain only original images. The dataset location is set by `DATASET_DIR` at the top of the script.

| Split      | Images |
|------------|-------:|
| Train      | 321 (107 original) |
| Validation | 12     |
| Test       | 15     |

## Training and evaluation

All training and evaluation steps are in `Notebook_GlassesDet.ipynb`:

1. Dataset integrity checks
2. Exploratory data analysis: split sizes, class balance, image resolutions, bounding-box statistics
3. Visual inspection of ground-truth annotations
4. Fine-tuning of YOLO12m
5. Evaluation on the validation and test sets, both overall and per class
6. Confusion matrices, precision-recall and F1 curves, and sample predictions

Training configuration:

| Parameter      | Value                              |
|----------------|------------------------------------|
| Base model     | YOLO12m (pretrained)               |
| Image size     | 640                                |
| Batch size     | 8                                  |
| Epochs         | 100 max, early stopping patience 20 |
| Optimizer      | AdamW, lr 0.001                    |
| Seed           | 42                                 |

The best checkpoint is saved to `runs/detect/runs/yolo12m_glassesdet_notebook/weights/best.pt`.

## Webcam demo

```bash
uv run python live_inference.py
```

The demo runs the trained model on the default webcam and shows the detections along with a live FPS counter. Press `q` to quit. The model path, camera index, confidence threshold (0.30 by default) and input size are set at the top of the script.

## Limitations

- The dataset is small. Augmentation adds variation but no genuinely new samples.
- With so few validation and test images, a single prediction can shift the metrics noticeably.
- The webcam demo is only a qualitative check and is not included in the reported metrics.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Glasses Detection (YOLO format)](https://www.kaggle.com/datasets/mohamedchahed/glasses-detection-yolo-format) dataset by mohamedchahed
- [Albumentations](https://albumentations.ai/)
