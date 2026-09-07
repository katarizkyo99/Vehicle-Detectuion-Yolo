```markdown
# Vehicle Type Detection (YOLOv8 + Streamlit)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vehicle-detection-yolo-ocdwr4afznns6xwfsgd7na.streamlit.app/)

**Live Demo:** [Vehicle Detection Web App](https://vehicle-detection-yolo-ocdwr4afznns6xwfsgd7na.streamlit.app/)

A computer vision web application built entirely with Streamlit to detect and classify vehicle types (car, motorcycle, bus, truck) in uploaded images. The application utilizes a YOLOv8 model for object detection and the Supervision library for clean, professional bounding box annotations. 

## Features

* **Interactive UI:** Upload images directly through a clean Streamlit interface.
* **Real-time Inference:** Fast vehicle detection and classification using Ultralytics YOLOv8.
* **Advanced Annotation:** Custom bounding boxes and confidence score labels rendered via Supervision.
* **In-Memory Processing:** Images are processed entirely in RAM without requiring local disk storage for uploads or results.
* **Cloud-Ready:** Deployed seamlessly on Streamlit Community Cloud with pre-configured system graphics dependencies.

## Tech Stack

* **Frontend & Backend**: Python, Streamlit
* **Computer Vision**: Ultralytics (YOLOv8)
* **Image Processing**: OpenCV (`opencv-python-headless`), Numpy, Pillow
* **Annotation**: Supervision
* **Dataset Management**: Roboflow (Optional integration included)

## Project Structure

```text
project/
├── main.py              # Core Streamlit app and inference pipeline
├── requirements.txt     # Python dependencies
├── packages.txt         # Linux system dependencies (libgl1) for cloud deployment
├── foto/                # Sample testing images
└── README.md

```

## Requirements

* Python 3.8+
* `pip`
* Required Python packages: `streamlit`, `ultralytics`, `supervision`, `opencv-python-headless`, `Pillow`, `numpy`, `roboflow`.

## Installation & Local Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/katarizkyo99/Computer-Vision_Deteksi-Jenis-Kendaraan.git](https://github.com/katarizkyo99/Computer-Vision_Deteksi-Jenis-Kendaraan.git)
cd Computer-Vision_Deteksi-Jenis-Kendaraan

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## Configuration

You can configure the application using environment variables to keep your API keys secure and manage model paths dynamically:

* `YOLO_WEIGHTS`: Set the path to your custom `.pt` file. If unset, the app defaults to `yolov8x.pt`.
* `ROBOFLOW_API_KEY`: Set your Roboflow API key if you need to download datasets programmatically.

Example configuration (Linux/macOS):

```bash
export YOLO_WEIGHTS="runs/detect/train/weights/best.pt"
export ROBOFLOW_API_KEY="your_api_key_here"

```

## Running the Project Locally

Start the Streamlit server:

```bash
streamlit run main.py

```

The application will automatically open in your default web browser at `http://localhost:8501`.

## Cloud Deployment

This project is actively deployed on Streamlit Community Cloud.

To ensure OpenCV works correctly in a headless Linux server environment (like Streamlit Cloud), this repository includes a `packages.txt` file containing `libgl1`. This resolves the common `libGL.so.1` missing dependency error.

## Author

* **Repository Owner:** [katarizkyo99](https://www.google.com/search?q=https://github.com/katarizkyo99)

```

```
