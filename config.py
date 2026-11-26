# Camera-specific approach regions. Coordinates are normalized (0..1).
# Edit these polygons after viewing your own camera frame.
# Each polygon should cover the incoming queue/approach for that direction.
LANE_POLYGONS = {
    "North": [(0.35, 0.00), (0.65, 0.00), (0.58, 0.43), (0.42, 0.43)],
    "South": [(0.42, 0.57), (0.58, 0.57), (0.65, 1.00), (0.35, 1.00)],
    "East":  [(0.57, 0.35), (1.00, 0.35), (1.00, 0.65), (0.57, 0.58)],
    "West":  [(0.00, 0.35), (0.43, 0.42), (0.43, 0.58), (0.00, 0.65)],
}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
EMERGENCY_CLASSES = {"ambulance", "fire_truck", "police_vehicle"}
