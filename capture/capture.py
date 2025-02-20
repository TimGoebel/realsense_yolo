# capture.py
import pyrealsense2 as rs
import numpy as np
import cv2
import queue

def capture_frames(pipeline, align, colorizer, filters, depth_scale, stop_event, frame_queue):
    threshold_filter, disparity_transformer, spatial_filter, temporal_filter, disparity_to_depth, hole_filling = filters
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        depth_data = np.asanyarray(depth_frame.get_data())
        value_min = max(0.2, depth_data.min() * depth_scale)
        value_max = min(5.0, depth_data.max() * depth_scale)
        colorizer.set_option(rs.option.min_distance, value_min)
        colorizer.set_option(rs.option.max_distance, value_max)
        depth_frame_aligned = depth_frame

        # Update and apply filters
        threshold_filter.set_option(rs.option.min_distance, value_min)
        threshold_filter.set_option(rs.option.max_distance, value_max)
        depth_frame_aligned = threshold_filter.process(depth_frame_aligned)
        depth_frame_aligned = disparity_transformer.process(depth_frame_aligned)
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        spatial_filter.set_option(rs.option.holes_fill, 0)
        depth_frame_aligned = spatial_filter.process(depth_frame_aligned)
        temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        depth_frame_aligned = temporal_filter.process(depth_frame_aligned)
        depth_frame_aligned = disparity_to_depth.process(depth_frame_aligned)
        depth_frame_aligned = hole_filling.process(depth_frame_aligned)
        depth_frame_aligned1 = depth_frame_aligned.as_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame_aligned).get_data())

        try:
            frame_queue.put((color_image, depth_colormap, intrinsics, depth_frame_aligned1), timeout=0.1)
        except queue.Full:
            pass
