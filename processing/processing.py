# processing.py
import cv2
import numpy as np
import queue
from utils.utils import crop_zone_process

def process_frames(model, set_points, depth_scale, stop_event, frame_queue, result_queue):
    while not stop_event.is_set():
        try:
            color_image, depth_colormap, intrinsics, depth_frame_aligned1 = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        processed_color, processed_depth = crop_zone_process(
            color_image, depth_colormap, model, set_points,
            depth_frame_aligned1, depth_scale, intrinsics
        )
        images = np.hstack((processed_color, processed_depth))
        try:
            result_queue.put(images, timeout=0.1)
        except queue.Full:
            pass
