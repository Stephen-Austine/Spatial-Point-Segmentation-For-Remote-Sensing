# Point-Supervised Remote Sensing Segmentation

This repository contains an end-to-end PyTorch notebook for training a point-supervised remote sensing scene segmentation model. The project uses sparse point labels instead of full pixel masks and compares multiple point-sampling densities and loss formulations for a four-class land/water/terrain scene classification problem.

The work is implemented in the notebook:

- `point_seg_assessment_Stephen_Austine.ipynb`

The generated model artifact:

- `point_seg_net_main.pth`

and supporting image outputs:

- `point_annotations.png`
- `exp1_point_density.png`
- `exp2_loss_comparison.png`
- `qualitative_results.png`

## Project Overview

This project builds a segmentation model that takes a remote sensing image and outputs a dense pixel-wise class map over four scene categories:

- `cloudy`
- `desert`
- `green_area`
- `water`

The key idea is point supervision:

1. The dataset only has image-level scene labels for each image.
2. The image-level class is treated as a pseudo-label for every pixel in the image.
3. A small random fraction of the pixels are sampled as sparse point labels.
4. The remaining pixels are marked as unlabeled and ignored by the loss.

That setup turns a dataset without ground-truth segmentation masks into a weakly supervised segmentation pipeline.

## Dataset

The image data are stored under the `data/` directory, organized by class folder:

- `data/cloudy/`
- `data/desert/`
- `data/green_area/`
- `data/water/`

The image source is the public dataset referenced in the notebook:

https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

The project uses image files resized to `64 × 64` and normalized using ImageNet mean and standard deviation values.

The class mapping used by the notebook is:

```python
CLASS_NAMES = ["cloudy", "desert", "green_area", "water"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
```

The notebook also creates a random 80/20 train/validation split.

## Core Training Setup

The notebook uses the following configuration:

```python
DATA_ROOT = "./data"
IMG_SIZE = 64
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
DEFAULT_POINT_RATE = 0.10
```

The default point rate is 10% of the image pixels.

## Data Pipeline

A custom `RemoteSensingDataset` class implements the point-sampling pipeline:

- Each image is loaded from disk.
- The image is resized to `64 × 64`.
- If augmentation is enabled, simple horizontal/vertical flipping and color jitter are applied.
- A tensor transform normalizes the image.
- A `point_mask` selects a random subset of pixels as supervised points.
- A sparse target tensor is created where only the sampled points receive the image-level class label, and every other pixel is marked with `-1`.

The data example returned by the dataset is:

```python
(img_tensor, target, point_mask, label)
```

where:

- `img_tensor`: normalized image tensor
- `target`: full-resolution sparse target map with `-1` on unlabeled pixels
- `point_mask`: boolean mask that identifies labeled points
- `label`: scene class label that is used as the pseudo-class for the sampled pixels

## Loss Functions

The notebook implements a custom `PartialCrossEntropyLoss` and compares it against a vanilla standard cross entropy baseline.

### Partial Cross-Entropy

The loss is computed only on labeled pixels, and the final average is taken over the sampled point mask, not over the entire image.

For a label mask `M`:

$$
\mathcal{L}_{pCE} = \frac{1}{|M|} \sum_{i \in M} CE(logits_i, y_i)
$$

where `M` is the set of pixels where a point label exists.

The notebook also supports an optional focal variant:

$$
\mathcal{L}_{focal} = (1 - p_t)^\gamma \cdot CE
$$

with `gamma = 2` in the experiment section.

The notebook implements the following losses:

1. `PartialCrossEntropyLoss(focal_gamma=0.0)`
2. `PartialCrossEntropyLoss(focal_gamma=2.0)`
3. `StandardCEWithIgnore()` using `ignore_index=-1`

## Model Architecture

The segmentation model is named `PointSegNet` and uses a pretrained `ResNet-18` encoder.

The architecture is:

```text
3 x 64 x 64
    -> ResNet-18 encoder (truncated after layer 3)
    -> decoder
    -> 4 x 64 x 64 segmentation logits
```

The encoder path is intentionally stopped at layer 3 so that the model retains more spatial detail. The decoder is a lightweight convolutional head:

- `Conv2d(256, 128, 3x3)`
- `BatchNorm2d(128)`
- `ReLU`
- `Conv2d(128, 64, 3x3)`
- `BatchNorm2d(64)`
- `ReLU`
- `Conv2d(64, num_classes, 1x1)`

It then uses bilinear interpolation to resize the segmentation output back to the input image shape, yielding a dense `4 × 64 × 64` map.

## Training Procedure

The training loop is defined in the notebook:

- `train_one_epoch()` updates model weights.
- `evaluate()` computes loss, point accuracy, and image-level accuracy.
- `run_experiment()` creates data loaders, instantiates `PointSegNet`, configures the optimizer and scheduler, and runs training.

The optimizer is:

```python
torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

The scheduler is:

```python
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

The training code uses the device that is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Experiments

### Experiment 1: Point Density

The notebook studies how performance changes when the sparse point label rate varies:

- 1%
- 5%
- 10%
- 20%

This comparison tests whether more point supervision helps the model produce better point accuracy and image-level accuracy.

### Experiment 2: Loss Function Comparison

The notebook compares:

- Standard CE with ignored background/unlabeled pixels via `ignore_index=-1`
- Partial CE with `gamma=0`
- Partial CE with focal weighting `gamma=2`

These variants are evaluated under the same point rate and training settings.

## Evaluation Metrics

The evaluation function records:

- `val_loss`
- `val_pt_acc`: point-level accuracy on labeled pixels
- `val_img_acc`: image-level accuracy derived from the modal predicted class in the dense segmentation output

The notebook’s evaluation logic:

1. Runs a forward pass through the model.
2. Applies the partial cross-entropy loss on the sampled point labels.
3. Computes point accuracy by comparing predictions only on `point_mask` pixels.
4. Computes image-level accuracy by taking the most common predicted class in each image segmentation map and comparing it to the image’s class label.

## Generated Artifacts

The repository contains several generated figures:

- `point_annotations.png`: sampled synthetic point labels over representative scene images.
- `exp1_point_density.png`: experiment plot for point density.
- `exp2_loss_comparison.png`: experiment plot for loss ablations.
- `qualitative_results.png`: sample input images and colored predicted segmentation maps.

The trained model checkpoint is saved in the workspace as:

- `point_seg_net_main.pth`

## Repository Structure

```text
point_seg_assessment/
├── data/
│   ├── cloudy/
│   ├── desert/
│   ├── green_area/
│   └── water/
├── point_seg_assessment_Stephen_Austine.ipynb
├── point_seg_net_main.pth
├── point_annotations.png
├── exp1_point_density.png
├── exp2_loss_comparison.png
├── qualitative_results.png
└── technical_report_Stephen_Austine.pdf
```

## Environment

The notebook depends on:

- Python
- PyTorch
- Torchvision
- PIL / Pillow
- NumPy
- Matplotlib

A local environment should support the notebook’s imports with the packages listed above.

## How to Run

To reproduce the project:

1. Open the notebook:

```text
point_seg_assessment_Stephen_Austine.ipynb
```

2. Ensure the image folders under `data/` are present.

3. Run the notebook cells in order from the top.

4. Optionally, reload the trained model from `point_seg_net_main.pth` for inference.

You can also inspect the generated figures in the root directory.

## Notes

This project is a weak-label semantic segmentation experiment in remote sensing. It differs from a fully supervised segmentation workflow because it does not require per-pixel labels or dense segmentation masks.

Instead, it uses sparse point labels generated from image-level labels, enabling a practical segmentation pipeline for data collections that contain only scene labels.

## Summary

This repository demonstrates a compact, reproducible PyTorch workflow for point-supervised semantic segmentation on remote sensing image categories. The structure is centered around a notebook that implements data loading, synthetic point generation, loss design, model architecture, training, evaluation, experiments, and qualitative visualization.
