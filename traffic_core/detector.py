import cv2
from ultralytics import YOLO
from config import VEHICLE_CLASSES, EMERGENCY_CLASSES
from traffic_core.geometry import find_approach

class TrafficDetector:
    def __init__(self, vehicle_weights="yolov8n.pt", emergency_weights=None, confidence=.35):
        self.vehicle_model = YOLO(vehicle_weights)
        self.emergency_model = YOLO(emergency_weights) if emergency_weights else None
        self.confidence = confidence

    def process(self, frame):
        h, w = frame.shape[:2]
        result = self.vehicle_model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=self.confidence, verbose=False
        )[0]
        counts = {lane: 0 for lane in ["North", "South", "East", "West"]}
        unique = []
        emergency_lanes = []
        annotated = frame.copy()

        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes
            coords = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(coords)
            for xyxy, cls_id, track_id in zip(coords, classes, ids):
                label = result.names[int(cls_id)]
                if label not in VEHICLE_CLASSES:
                    continue
                x1,y1,x2,y2 = xyxy.astype(int)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                lane = find_approach(cx, cy, w, h)
                if lane:
                    counts[lane] += 1
                unique.append({"id": int(track_id), "class": label, "lane": lane})
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(30,220,30),2)
                cv2.putText(annotated,f"{label} #{track_id}",(x1,max(18,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(30,220,30),1)

        if self.emergency_model is not None:
            er = self.emergency_model.predict(frame, conf=max(.25,self.confidence), verbose=False)[0]
            if er.boxes is not None:
                for box in er.boxes:
                    cls_id = int(box.cls.item())
                    label = str(er.names[cls_id]).lower()
                    confidence = float(box.conf.item())
                    if label not in EMERGENCY_CLASSES:
                        continue
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    lane = find_approach((x1+x2)//2,(y1+y2)//2,w,h)
                    print(
    f"Emergency: {label}, "
    f"Approach: {lane}, "
    f"Confidence: {confidence:.2f}"
)
                    if lane:
                        emergency_lanes.append(lane)
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(annotated,f"EMERGENCY: {label}",(x1,max(20,y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        return annotated, counts, unique, emergency_lanes
