# RealSense YOLO on Jetson Pipeline 📷 🟩

This project implements a RealSense-based pipeline that captures frames from an Intel RealSense camera, applies depth filtering, and performs YOLO-based object detection all on a Jetson. The application leverages asynchronous processing (using Python's threading and queue modules) to separate frame capture from heavy inference tasks, improving performance and making troubleshooting easier.

## Features

- **RealSense Integration:**  
  Captures depth and color data from an Intel RealSense camera.
- **Depth Filtering:**  
  Applies a series of filters (threshold, disparity transformation, spatial, temporal, and hole-filling) to enhance depth data.
- **YOLO Object Detection:**  
  Performs object detection using an Ultralytics YOLO model.
- **Asynchronous Processing:**  
  Uses separate threads for frame capture and YOLO inference to decouple heavy processing from frame acquisition.
- **Modular Structure:**  
  Organized into multiple modules for easier troubleshooting and maintenance.

## File Structure
```
📁 realsense_yolo
│── 📄 main.py          # Entry point for the application.
│── 📁 capture
│   │── 📄 capture.py# Module handling RealSense frame capture and filtering.
│── 📁 processing
│   │── 📄 processing.py    # Module for performing YOLO inference on captured frames.
├── processing.py    # Module for performing YOLO inference on captured frames.
│── 📁 utils
│   │── 📄 utils.py         # Utility functions (configuration loading, YOLO processing, etc.).
│── 📁 models
│   │── 📄 model.pt         # put your model there
└── config_m.json    # Configuration file with settings and model paths.
```

## Requirements

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Install the dependencies via pip:
```sh
pip install pyrealsense2 opencv-python ultralytics numpy
```

## Configuration
The `config_m.json` file should include settings for the model path, confidence threshold, output video path, and frame dimensions. An example configuration file:
```json
{
  "model": "path/to/model/yolov11.weights",
  "CONFID_THRESHOLD": 50,
  "output_video_path": "output.mp4",
  "frame_width_record": 1280,
  "frame_height_record": 480
}
```
Ensure the model path and other parameters are correct for your setup.

## Usage

### 1. Clone the Repository:
```sh
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Run the Application:
```sh
python main.py
```
The RealSense pipeline will start, processing frames asynchronously while displaying combined color and depth images. Press `q` to exit the application.

## Troubleshooting

### Frame Not Loading:
Ensure your RealSense camera is properly connected and the necessary drivers are installed.

### QObject Timer Error:
If you encounter errors like `QObject::startTimer: Timers cannot be started from another thread`, make sure that GUI operations (such as `cv2.imshow` and `cv2.waitKey`) run in the main thread.

### Performance Issues:
Adjust queue sizes, timeout values, or the frequency of YOLO inference to balance throughput and latency. Profiling the code may also help identify bottlenecks.

### Queue Size:

Adjust the maximum size of the frame and result queues. A smaller queue may reduce latency but can cause frame drops if processing can’t keep up. Conversely, a larger queue might buffer more frames but add delay. Experiment with values (e.g., 2–5).

### Timeout Values:

Reduce the timeout values in queue.put() and queue.get() calls so threads don’t block longer than necessary. For example, if you’re currently using 0.1 seconds, try a lower value (e.g., 0.05).

## Contributing
Contributions are welcome! If you find bugs or have suggestions for improvements, please open an issue or submit a pull request.

## Note
The models folder the file in there needs to be replace with your file

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
🚀 Built for AI-driven computer vision applications!
