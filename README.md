# DiffusionDet: Diffusion-based Object Detection

This repository implements a diffusion-based object detection model trained on a subset of the COCO dataset. The model uses a diffusion process to refine object proposals and predict bounding boxes and class labels.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gerceboss/DiffDet
cd diffusiondet
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv diff_venv
source diff_venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip3 install torch torchvision torchaudio
pip3 install pyyaml tqdm matplotlib
pip3 install pycocotools
```

## Dataset Setup

The model uses a subset of the COCO 2017 validation dataset, focusing on three classes:
- Person (class 0)
- Car (class 1)
- Dog (class 2)

To download and prepare the dataset:
```bash
python3 dataset/download_coco.py
```

This will:
- Download COCO 2017 validation annotations
- Download 100 images containing the target classes
- Create `dataset/` directory with:
  - `images/`: Downloaded COCO images
  - `annotations.json`: Processed annotations

## Training

To train the model:
```bash
python3 main.py train
```

Training configuration is in `configs/config.yaml`:
- `num_classes`: 3 (person, car, dog)
- `num_images`: 100
- `batch_size`: 8
- `epochs`: 50
- `num_proposals`: 50
- `timesteps`: 1000
- `lr`: 1e-4
- `image_size`: 224

Training outputs:
- `weights/diffusiondet.pth`: Trained model weights
- `training_loss.png`: Training loss curve
- `training.log`: Detailed training logs

## Evaluation

To evaluate the model:
```bash
python3 main.py eval
```

valuation outputs:
- `eval_result_0.png` to `eval_result_4.png`: Visualization of predictions on 5 test images
  - Green boxes: Ground truth annotations
  - Colored boxes: Model predictions
    - Red: Person
    - Blue: Car
    - Yellow: Dog

## Results Interpretation

1. **Training Loss Plot** (`training_loss.png`):
   - X-axis: Training epochs
   - Y-axis: Loss value
   - Lower values indicate better training

2. **Evaluation Images** (`eval_result_*.png`):
   - Each image shows both ground truth and predictions
   - Box colors indicate different classes
   - Box transparency indicates prediction confidence
   - Good results show:
     - High overlap between green (ground truth) and colored (predicted) boxes
     - Correct class predictions (matching colors)
     - Boxes distributed across the image, not just corners

NOTE: This is a very lightweight version of original DiffusionDet model