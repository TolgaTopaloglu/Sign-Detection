# Sign Detection with YOLO11

This project is a deep learning-based object detection system that detects traffic and street signs using a model trained with Ultralytics YOLO11. The model was trained on a dataset imported from Roboflow containing 7,796 images across 26 classes. The trained model is used in Python for inference on images to identify and localize signs using bounding box predictions. The main goal of this project is to demonstrate a real-world application of object detection using a custom-trained dataset.

## Objectives
- Train a custom object detection model for sign recognition.
- Detect signs in static images.
- Visualize bounding boxes on detected objects.
- Run inference using a trained YOLO model.
- Build a simple and efficient Python-based pipeline.

## Dataset
- **Name:** TabelaTanıma
- **Source:** [Roboflow Universe](https://universe.roboflow.com/deneme-o9iiw/tabelatanima-jg4kg/dataset/1)
- **Size:** 7,796 images, 26 classes
- **Format:** YOLO

## Training

The model was fine-tuned from the pretrained `yolo11s.pt` on Google Colab (GPU).

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=1024,
    batch=32,
    device=0,
    workers=8,
    patience=20,
    fliplr=0.0,
    flipud=0.0,
    degrees=0.0,
    seed=42,
)
```

| Parameter | Value | Why |
|---|---|---|
| Base model | `yolo11s` | Good balance between speed and accuracy |
| `imgsz` | 1024 | Signs are often small in the frame; a larger input helps detect them |
| `epochs` / `patience` | 100 / 20 | Upper limit on epochs; early stopping if validation metrics stop improving for 20 epochs |
| `batch` | 32 | |
| `fliplr` | 0.0 | Horizontal flipping is disabled: mirroring changes a sign's meaning (e.g. "no left turn" ↔ "no right turn") and reverses text |
| `flipud`, `degrees` | 0.0 | No vertical flips or rotations; signs appear upright in real scenes |
| `seed` | 42 | Reproducibility |

## Results

Validation metrics at the last epoch of the training run shown below (19 epochs; approximate values read from the curves):

| Metric | Value |
|---|---|
| Precision | ≈ 0.90 |
| Recall | ≈ 0.93 |
| mAP50 | ≈ 0.95 |
| mAP50-95 | ≈ 0.76 |

<img width="1000" height="500" alt="Precision, recall and mAP per epoch" src="https://github.com/user-attachments/assets/9acc8e60-7956-4dbe-81db-9c89037b31db" />

<img width="1500" height="500" alt="Train and validation losses per epoch" src="https://github.com/user-attachments/assets/13d12cf2-0565-46a9-b33f-09ced6f2a399" />

<img width="3000" height="2250" alt="Confusion matrix" src="https://github.com/user-attachments/assets/53cdfd1a-8ff1-4cec-a414-58e62b212c68" />

## Usage

```python
from ultralytics import YOLO

model = YOLO("TabelaTanıma.pt")
model.predict("image.jpg", imgsz=1024, save=True)
```

## Files

| File | Description |
|---|---|
| `TabelaTanıma.pt` | Trained weights (`best.pt` from training) |
| `Çıktı*.png/jpg` | Sample detection outputs |
| `requirements.txt` | Python dependencies |
