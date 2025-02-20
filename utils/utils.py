# utils.py
import os
import json
import numpy as np
import cv2
import math
from datetime import datetime

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6

def read_json_config():
    cwd = os.getcwd() 
    json_path = os.path.join(cwd, "config_m.json")
    with open(json_path, "r") as json_file:
        config = json.load(json_file)
    return config

def crop_zone_process(color_frame, depth_colormap, model, set_points, depth_frame_aligned, depth_scale, intrinsics):
    # YOLO detection on the cropped region
    results = model(color_frame)
    for info in results:
        for box in info.boxes:
            confidence = int(box.conf[0] * 100)
            # Only proceed if confidence passes threshold
            if confidence > int(set_points.get("CONFID_THRESHOLD")):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # The rest of your 3D drawing and measurement logic...
                edge_pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                vertical_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
                top_center = [(0,1), (2,3)]
                line_color = (0, 255, 255)
                line_thickness = 2
                xy_np = info.keypoints.xy.cpu().numpy()
                for pair in edge_pairs:
                    i, j = pair
                    if xy_np[0][i][0] == 0 and xy_np[0][i][1] == 0:
                        continue
                    if xy_np[0][j][0] == 0 and xy_np[0][j][1] == 0:
                        continue

                    cv2.line(depth_colormap, tuple(map(int, xy_np[0][i])),
                             tuple(map(int, xy_np[0][j])), line_color, thickness=line_thickness)
                    cv2.circle(depth_colormap, tuple(map(int, xy_np[0][i])), 2, (0, 0, 255), -1)
                    cv2.putText(depth_colormap, str(i), tuple(map(int, xy_np[0][i])),
                                fontScale=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                color=(255, 255, 255), thickness=2)
                    # (Add your 3D coordinate conversion and distance measurement here)
    return color_frame, depth_colormap
