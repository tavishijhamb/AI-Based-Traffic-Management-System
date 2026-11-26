from ultralytics import YOLO
from pathlib import Path

DATASET = Path(
    r"C:\Users\Tavishi\Downloads\dataset1\data.yaml"
)

def main():

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATASET),
        epochs=30,
        imgsz=640,
        batch=8,
        patience=7,
        project="runs/emergency",
        name="traffic_emergency_30",
        plots=True,
        seed=42,
        save=True
    )

    print("Training completed!")

if __name__ == "__main__":
    main()