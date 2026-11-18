# Laser-Guided Pick-to-Light System for Smart Manufacturing

YOLOv8-based parts detection with laser-pointer guidance and hand-feedback for a smart manufacturing workstation.

## Layout

- `Smart_Manufacturing_Project/` – main project
  - `scripts/` – training, webcam detection, counting, laser calibration/control, hand-feedback experiments
  - `laser_calibration.npy`, `mapping_model.npy` – saved calibration/mapping data
  - `runs/` – YOLO training results and weights
- `Smart_Manufacturing_Project2/` – second iteration (training + webcam detection)

## Datasets

The `dataset_raw/` and `dataset_split/` image folders (~1.9 GB) are not included in this repository.
Place them inside each project folder before running `split_dataset.py` / `train.py`.
