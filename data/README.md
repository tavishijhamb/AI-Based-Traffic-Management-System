# Dataset and annotation guide

## Vehicle detection
The app starts with pretrained YOLOv8n COCO weights. The standard COCO classes include car, motorcycle, bus and truck. This is enough to test the end-to-end video pipeline.

For fine-tuning, use:
- AI City Challenge traffic videos/counting data: https://www.aicitychallenge.org/ai-city-challenge-dataset-access/
- BDD100K: https://github.com/bdd100k/bdd100k
- UA-DETRAC: https://detrac-db.rit.albany.edu/

Convert annotations to YOLO format if needed. Keep train/validation/test split grouped by source video or camera, not adjacent random frames.

## Emergency vehicle dataset
Prepare this structure:
```
data/emergency/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```
YOLO label line format: `class_id x_center y_center width height`, with normalized coordinates 0..1.
Use consistent classes: ambulance=0, fire_truck=1, police_vehicle=2.
Example data.yaml:
```
path: data/emergency
train: images/train
val: images/val
names:
  0: ambulance
  1: fire_truck
  2: police_vehicle
```
Include ordinary vehicles (especially vans, white trucks and similar-looking vehicles) as negative examples. Verify licensing and annotation quality. Split by video/source to avoid leakage.

## Congestion history
The dashboard writes observations to data/traffic_history.csv every ~10 seconds, with columns:
timestamp,lane,vehicle_count,minute_of_day,day_of_week,density
Collect data across multiple days and traffic conditions. The starter model learns to classify the current density from count/time features; it is not a sophisticated future time-series forecast. For a proper future forecast, add lagged counts for previous intervals and create a target such as density 5 or 10 minutes ahead. Use chronological train/test splits.
