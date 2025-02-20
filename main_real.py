# main_real.py
import os
import cv2
import threading
import queue
import pyrealsense2 as rs
from ultralytics import YOLO
from utils.utils import read_json_config
from capture.capture import capture_frames
from processing.processing import process_frames

def main():
    config = read_json_config()
    model = YOLO(config.get("model"))

    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        pipeline.start(config_rs)
        print("Pipeline started successfully.")
    except RuntimeError as e:
        print("Failed to start pipeline:", e)
        return

    align_to = rs.stream.color
    align = rs.align(align_to)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 9)
    colorizer.set_option(rs.option.visual_preset, 0)
    colorizer.set_option(rs.option.histogram_equalization_enabled, 1)

    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    # Instantiate filters once
    threshold_filter = rs.threshold_filter()
    disparity_transformer = rs.disparity_transform(True)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter(1)
    filters = (threshold_filter, disparity_transformer, spatial_filter,
               temporal_filter, disparity_to_depth, hole_filling)

    # Create queues and a stop event
    frame_queue = queue.Queue(maxsize=5)
    result_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # Start capture and processing threads
    capture_thread = threading.Thread(
        target=capture_frames,
        args=(pipeline, align, colorizer, filters, depth_scale, stop_event, frame_queue)
    )
    process_thread = threading.Thread(
        target=process_frames,
        args=(model, config, depth_scale, stop_event, frame_queue, result_queue)
    )
    capture_thread.start()
    process_thread.start()

    # Main thread: display frames from the result queue
    try:
        while not stop_event.is_set():
            try:
                images = result_queue.get(timeout=0.1)
                cv2.imshow('RealSense combine', images)
            except queue.Empty:
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        capture_thread.join()
        process_thread.join()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
