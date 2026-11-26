import cv2
import numpy as np
from config import LANE_POLYGONS

def polygon_pixels(points, width, height):
    return np.array([(int(x * width), int(y * height)) for x, y in points], dtype=np.int32)

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def find_approach(cx, cy, width, height):
    """Assign a detection centre to a configured approach polygon."""
    for name, points in LANE_POLYGONS.items():
        poly = polygon_pixels(points, width, height)
        if point_in_polygon((int(cx), int(cy)), poly):
            return name
    return None

def draw_approaches(frame):
    h, w = frame.shape[:2]
    out = frame.copy()
    for name, points in LANE_POLYGONS.items():
        poly = polygon_pixels(points, w, h)
        cv2.polylines(out, [poly], True, (255, 180, 0), 2)
        x, y = poly[0]
        cv2.putText(out, name, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,180,0), 2)
    return out
