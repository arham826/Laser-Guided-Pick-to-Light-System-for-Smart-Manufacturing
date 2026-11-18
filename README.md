# Laser-Guided Pick-to-Light System for Smart Manufacturing

A vision-based **pick-to-light** assistant for manual assembly stations. A camera watches a tray of parts, a YOLOv8 model detects all 17 part types (`p1`–`p17`) plus the operator's **hand**, and a pan/tilt laser driven by an Arduino points at the next part the operator has to pick. When the system sees the hand pick up the highlighted part, it moves on to the next part in the assembly sequence.

Most pick-to-light systems need a fixed LED on every bin. This one uses only a camera and a laser, so parts can sit anywhere on the work surface and bins can be moved around without rewiring.

---

## Table of contents

- [How it works](#how-it-works)
- [Hardware](#hardware)
- [Software requirements](#software-requirements)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Training the detector](#training-the-detector)
- [Model performance](#model-performance)
- [Laser calibration](#laser-calibration)
- [Running the pick-to-light system](#running-the-pick-to-light-system)
- [Utility and experimental scripts](#utility-and-experimental-scripts)
- [Arduino serial protocol](#arduino-serial-protocol)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future work](#future-work)

---

## How it works

```
 ┌──────────┐   frames   ┌──────────────────┐  part boxes + hand box  ┌────────────────────┐
 │  Camera  │ ─────────▶ │  YOLOv8n detector │ ──────────────────────▶ │ Pick-sequence logic │
 └──────────┘            └──────────────────┘                          └─────────┬──────────┘
                                                                                 │ target pixel (cx, cy)
                                                                                 ▼
                                                               ┌─────────────────────────────────┐
                                                               │ Pixel → servo mapping            │
                                                               │ servoX = mx·px + bx              │
                                                               │ servoY = my·py + by              │
                                                               └────────────────┬────────────────┘
                                                                                │ "X,Y\n" over serial
                                                                                ▼
                                                               ┌─────────────────────────────────┐
                                                               │ Arduino + pan/tilt servos + laser│
                                                               └─────────────────────────────────┘
```

1. **Detection:** each camera frame goes through a custom-trained YOLOv8n model that detects the 17 part classes and the operator's hand.
2. **Target selection:** the system follows a predefined pick sequence (for example `p1 → p14 → p9 → p2 → …`). The current target part is highlighted in green on screen.
3. **Laser guidance:** the centre of the target's bounding box is converted from pixel coordinates to servo angles using a linear mapping found during calibration. The angles go to the Arduino, which points the laser at the part.
4. **Pick confirmation (hand feedback):** while waiting, the system checks whether a detected part's centre falls inside the hand's bounding box. A pick counts only after:
   - the hand is detected with confidence > `HAND_CONF_THRESHOLD` (0.50),
   - the part is inside the hand for at least `HAND_REQUIRED_FRAMES` frames in a row (temporal filtering), and
   - at least `HAND_COOLDOWN` seconds (3 s) have passed since the last pick, so one grab can't count twice.
5. **Next step:** after a confirmed pick, the system moves to the next part. When the sequence finishes, it shows **ALL PARTS COMPLETED**.

---

## Hardware

| Component | Notes |
|---|---|
| USB webcam | Mounted above the work surface, looking down. Camera index `0` or `1` depending on the script. |
| Arduino (Uno/Nano or similar) | Receives servo angles over USB serial (`COM3`, 115200 baud by default). |
| 2 × hobby servos (e.g. SG90/MG90S) | Pan/tilt bracket: X = pan, Y = tilt. |
| Laser diode module | Mounted on the pan/tilt bracket. **Use a low-power (Class 2 or lower) laser and never point it at eyes.** |
| PC with NVIDIA GPU (optional) | Training was done on a GTX 1650 (`batch=4`, `device=0`). Inference also runs on CPU, just slower. |

> The Arduino firmware is not included in this repository. Any sketch that reads lines of the form `X,Y\n` and writes them to two servos will work. See [Arduino serial protocol](#arduino-serial-protocol).

---

## Software requirements

- Python 3.9+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- pySerial

```bash
pip install ultralytics opencv-python numpy pyserial
```

For GPU training, install a CUDA-enabled PyTorch build first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

---

## Repository structure

```
.
├── Smart_Manufacturing_Project/          # Main project: detection + laser + hand feedback
│   ├── scripts/
│   │   ├── split_dataset.py              # 70/20/10 train/val/test split
│   │   ├── train.py                      # YOLOv8n training
│   │   ├── webcam_detect.py              # Live detection preview
│   │   ├── counting.py                   # Live per-part counting
│   │   ├── laser_jog_control.py          # Manual servo jogging from the keyboard
│   │   ├── laser_calibration.py          # Early calibration experiments
│   │   ├── laser_calibration_final.py    # Final camera ↔ laser calibration → mapping_model.npy
│   │   ├── click_to_move_laser.py        # Click on the video to point the laser there
│   │   ├── laser_point_feedback.py       # Laser pointing with visual feedback
│   │   ├── laser_sequence_all_parts.py   # Visit every detected part in order
│   │   ├── dynamic_compartments.py       # Group nearby detections into compartments
│   │   ├── hand_feedback.py … hand_feedback8.py   # Iterations of the pick-to-light loop
│   │   └── test*.py                      # Prototypes and hardware tests
│   ├── dataset_split/dataset.yaml        # 17-class dataset config (images not included)
│   ├── runs/detect/runs/parts_detection*/  # Training results, plots, and weights
│   ├── laser_calibration.npy             # Early scale-based calibration (scaleX, scaleY)
│   ├── mapping_model.npy                 # Final linear mapping [mx, bx, my, by]
│   └── yolov8n.pt                        # Pretrained YOLOv8n base weights
│
└── Smart_Manufacturing_Project2/         # Second model iteration: adds the "hand" class
    ├── scripts/                          # split_dataset.py, train.py, webcam_detect.py
    ├── dataset_split/dataset.yaml        # 18-class dataset config (hand + p1–p17)
    ├── runs/detect/runs/parts_detection/ # Training results and weights (used by hand_feedback8.py)
    └── yolov8n.pt
```

---

## Dataset

The image datasets are **not included** in this repository because of their size (about 1.9 GB in total). Only the `dataset.yaml` configs are tracked.

| Project | Images | Classes |
|---|---|---|
| `Smart_Manufacturing_Project` | 424 | 17 (`p1`–`p17`) |
| `Smart_Manufacturing_Project2` | 634 | 18 (`hand`, `p1`–`p17`) |

Labels are in YOLO format (one `.txt` per image: `class cx cy w h`, normalised). The class order is alphabetical as exported from Roboflow (`p1, p10, p11, …, p17, p2, …, p9`). The scripts convert class indices back to part names with a lookup dictionary (`correct_part_order` / `roboflow_index_to_part`).

To rebuild the dataset, put your files here:

```
<project>/dataset_raw/images/*.jpg
<project>/dataset_raw/labels/*.txt
```

Then run:

```bash
cd Smart_Manufacturing_Project2
python scripts/split_dataset.py
```

This creates `dataset_split/images/{train,val,test}` and `dataset_split/labels/{train,val,test}` with a random 70 / 20 / 10 split.

---

## Training the detector

```bash
cd Smart_Manufacturing_Project2
python scripts/train.py
```

Key hyperparameters (`scripts/train.py`):

| Parameter | Value | Reason |
|---|---|---|
| Base model | `yolov8n.pt` | Small, fast model for real-time use |
| Epochs | 60 (early stopping `patience=20`) | |
| Image size | 640 | |
| Batch size | 4 | Fits in GTX 1650 memory |
| Optimizer | AdamW, `lr0=0.001` | |
| Augmentation | rotation ±10°, translate 0.1, scale 0.4, horizontal flip 0.5, mosaic 0.5 | |
| `workers` | 0 | Avoids Windows multiprocessing issues |

Results are saved to `runs/detect/runs/parts_detection*/`, with the best weights in `weights/best.pt`.

---

## Model performance

Final-epoch validation metrics from `results.csv`:

| Run | Classes | Epochs | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| `Smart_Manufacturing_Project/.../parts_detection` | 17 | 60 | 0.974 | 0.969 | 0.974 | 0.490 |
| `Smart_Manufacturing_Project/.../parts_detection9` | 17 | 69 / 80 | 0.965 | 0.964 | 0.964 | 0.501 |
| `Smart_Manufacturing_Project2/.../parts_detection` | 18 (with hand) | 60 | **0.971** | **0.971** | **0.975** | **0.535** |

The `Smart_Manufacturing_Project2` model detects the hand as well as the parts, and it is the model used by the final pick-to-light script. Confusion matrices, PR/F1 curves and sample predictions are in each run folder.

---

## Laser calibration

The laser and camera have to share a coordinate system. `laser_calibration_final.py` does this automatically:

1. It moves the servos through a 4 × 4 grid of known angles (X ∈ {68, 82, 95, 109}, Y ∈ {114, 122, 130, 138}).
2. At each point, it finds the red laser dot in the camera image with HSV thresholding (two red hue ranges) and takes the centroid of the largest contour.
3. It fits a separate least-squares linear model for each axis:

   ```
   servoX = mx · pixelX + bx
   servoY = my · pixelY + by
   ```

4. It saves `[mx, bx, my, by]` to `mapping_model.npy`.

```bash
cd Smart_Manufacturing_Project
python scripts/laser_calibration_final.py
```

Run calibration again whenever the camera or laser mount moves. You can check the result with `click_to_move_laser.py`: click anywhere in the video window and the laser should land on that spot.

---

## Running the pick-to-light system

The final version is **`Smart_Manufacturing_Project/scripts/hand_feedback8.py`**.

1. Connect the Arduino and the camera.
2. Edit the settings at the top of the script if needed (COM port, camera index, thresholds). The model and calibration paths are found automatically.
3. Edit `sequence` to set your assembly's pick order.
4. Run:

   ```bash
   python Smart_Manufacturing_Project/scripts/hand_feedback8.py
   ```

On-screen display:

- **Yellow text:** `TARGET PART: pX`, the part to pick next
- **Green box:** the target part
- **Blue boxes:** all other detected parts
- **Red box + "HAND DETECTED":** the operator's hand
- **"PART PICK CONFIRMED":** a part is currently inside the hand box

Press **`q`** to quit.

---

## Utility and experimental scripts

| Script | Purpose |
|---|---|
| `webcam_detect.py` | Live YOLO detection with corrected class names |
| `counting.py` | Live count of each part type, refreshed every second |
| `laser_jog_control.py` | Move the servos manually from the keyboard (1° steps) to explore the workspace |
| `click_to_move_laser.py` | Point the laser at any pixel by clicking, to check calibration |
| `laser_point_feedback.py` | Laser pointing with closed-loop visual feedback |
| `laser_sequence_all_parts.py` | Visit `p1 → p17` in order and pause at each part |
| `dynamic_compartments.py`, `test4dynamic.py`, `test5dynamic.py` | Group nearby detections (distance threshold) into dynamic compartments |
| `hand_feedback.py` … `hand_feedback7.py` | Earlier versions of the hand-confirmation logic |
| `test.py`, `test3.py`, `test_feedback.py`, `test2_feedback.py` | Hardware and serial tests and prototypes |

---

## Arduino serial protocol

The PC sends one ASCII line per move:

```
<servoX>,<servoY>\n
```

Example: `95,122\n`. Angles are integers in degrees (0–180). The usable workspace is about 55–130° on both axes. The final scripts use **115200 baud**. A few early scripts (`laser_sequence_all_parts.py`, `test.py`) use 9600.

Minimal example sketch:

```cpp
#include <Servo.h>
Servo sx, sy;

void setup() {
  Serial.begin(115200);
  sx.attach(9);   // pan
  sy.attach(10);  // tilt
  sx.write(90); sy.write(90);
}

void loop() {
  if (Serial.available()) {
    int x = Serial.parseInt();
    int y = Serial.parseInt();
    if (Serial.read() == '\n') {
      sx.write(constrain(x, 0, 180));
      sy.write(constrain(y, 0, 180));
    }
  }
}
```

---

## Configuration

**File paths are relative to each script's own location**, so the repository works from any folder or drive, and you can run a script from any working directory. Each script builds its paths from one of these:

```python
REPO_ROOT   = Path(__file__).resolve().parents[2]   # the repository root
PROJECT_DIR = Path(__file__).resolve().parents[1]   # the script's project folder
```

For example, `hand_feedback8.py` loads
`REPO_ROOT / "Smart_Manufacturing_Project2" / "runs" / "detect" / "runs" / "parts_detection" / "weights" / "best.pt"`. The calibration scripts always save their `.npy` files in `Smart_Manufacturing_Project/`, and `train.py` writes to `<project>/runs/detect/runs/`.

You may still need to change these hardware settings at the top of a script:

| Constant | Example |
|---|---|
| `PORT` / `COM_PORT` | `COM3` on Windows, `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux |
| `CAMERA_INDEX` | `0` or `1` |
| `PART_CONF_THRESHOLD`, `HAND_CONF_THRESHOLD` | `0.25`, `0.50` |
| `HAND_REQUIRED_FRAMES`, `HAND_COOLDOWN` | `2`, `3` s |

---

## Troubleshooting

- **`SerialException: could not open port`:** close the Arduino IDE Serial Monitor and check the COM port in Device Manager.
- **Black or no camera image:** try a different `CAMERA_INDEX`. On Windows, `cv2.VideoCapture(i, cv2.CAP_DSHOW)` often starts faster.
- **Wrong part names on screen:** make sure the class-index dictionary matches the model you loaded. The 17-class and 18-class models use different index orders.
- **Laser misses the part:** run `laser_calibration_final.py` again. The camera and laser mount must not move after calibration.
- **Laser dot not found during calibration:** dim the room lights or adjust the HSV thresholds in `detect_laser()`.
- **CUDA out of memory while training:** lower `batch` or `imgsz`.

---

## Future work

- Add the Arduino firmware to the repository
- Move hard-coded paths and settings into one config file
- Use a homography or polynomial mapping to handle lens distortion better than the per-axis linear model
- Load pick sequences from a file or an MES/ERP order
- Add wrong-pick detection with an audible or visual alert
- Log cycle time and pick accuracy for each operator
