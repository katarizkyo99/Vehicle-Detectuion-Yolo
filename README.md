# 🚦 TrafficLens — Real-Time Vehicle Detection & AI Traffic Analysis

<!-- Ganti baris di bawah dengan screenshot aplikasi kamu -->
![TrafficLens Screenshot](./assets/TrafficLens%20Interface%20Result.png)

A computer vision web app built with **Streamlit**, **YOLOv8 (Ultralytics)**, **Supervision**, and the **Groq Cloud API**. Upload any road/traffic image and TrafficLens detects, counts, and classifies vehicles (cars, motorcycles, buses, trucks), then generates a natural-language traffic composition analysis powered by an LLM.

---

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vehicle-object-detection-yolov8-qpw3b4wo2wsveqkzindrgh.streamlit.app/)
**Live Demo:** [TrafficLens]([https://your-trafficlens-app.streamlit.app/](https://vehicle-object-detection-yolov8-qpw3b4wo2wsveqkzindrgh.streamlit.app/))

## 🚀 Key Features

* **Upload Any Traffic Image:** Users upload their own road/traffic photos directly through the UI (`.jpg`, `.jpeg`, `.png`) — no fixed dataset required.
* **Configurable YOLOv8 Backbone:** Switch between four YOLOv8 checkpoints at runtime (`Nano`, `Small`, `Medium`, `X-Large`) to trade off speed vs. accuracy.
* **Adjustable Confidence Threshold:** Fine-tune detection sensitivity with a live slider (0.1–0.9).
* **Selectable Vehicle Classes:** Choose exactly which classes to detect (car, motorcycle, bus, truck, or any class the loaded model supports).
* **Annotated Output:** Bounding boxes and confidence-scored labels are drawn on the image using `supervision`, viewable side-by-side with the original via tabs.
* **Downloadable Results:** Export the annotated detection image as a PNG.
* **Composition Breakdown:** A live dashboard shows total vehicle count and per-class proportion bars.
* **AI-Powered Traffic Analysis:** A short, natural-language summary of the traffic composition is generated via Groq's LPU-accelerated inference.
* **Cached Model Loading:** YOLO weights are loaded once and cached via `@st.cache_resource` for fast repeated inference.

---

## 🛠 Tech Stack

* **Frontend / UI:** Streamlit
* **Object Detection:** Ultralytics YOLOv8
* **Detection Post-Processing & Annotation:** Supervision (`sv.BoxAnnotator`, `sv.LabelAnnotator`)
* **Image Processing:** OpenCV (`opencv-python-headless`), Pillow
* **LLM Engine:** Groq API (`ChatGroq` via `langchain-groq`)
* **Environment Management:** `python-dotenv`

---

## 📁 Repository Structure

```text
├── assets/                 # Screenshots and static assets for documentation
├── .env                    # Local environment variables (not committed)
├── .gitignore              # Git exclusion rules (e.g., .env, venv, __pycache__)
├── README.md               # Project documentation
├── main.py                 # Streamlit UI, YOLO inference & AI analysis pipeline
├── packages.txt            # System-level (apt) dependencies for Streamlit Cloud
└── requirements.txt        # Python dependencies
```

> Note: `packages.txt` is required specifically for deployment on Streamlit Community Cloud — it installs system libraries (like `libgl1`) that OpenCV needs but which aren't present on the base container image.

---

## 🧠 System Workflow

1. **Image Upload:** The user uploads a road/traffic image through the main uploader.
2. **Model Configuration:** The user selects a YOLOv8 checkpoint, confidence threshold, and target vehicle classes from the sidebar.
3. **Inference:** The image is passed to the loaded YOLOv8 model; results are converted into `sv.Detections` and filtered by the selected class IDs.
4. **Annotation:** Bounding boxes and confidence-labeled tags are drawn on a copy of the image via `supervision`.
5. **Aggregation:** Detected vehicles are grouped and counted per class to build the composition breakdown.
6. **AI Summary Generation:**

$$\text{Class Counts} \longrightarrow \text{Prompt Template} \longrightarrow \text{Groq LLM (ChatGroq)} \longrightarrow \text{Natural-Language Summary}$$

7. **Result Display:** The original and annotated images, composition bars, total count, and AI-generated analysis are rendered in the UI, with an option to download the annotated image.

---

## ⚙️ Installation & Local Setup

1. **Clone this repository:**
```bash
git clone https://github.com/<your-username>/TrafficLens.git
cd TrafficLens
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install system dependencies (Linux only, if `cv2` import fails locally):**
```bash
sudo apt-get update && sudo apt-get install -y $(cat packages.txt)
```

5. **Set up environment variables:**
Create a `.env` file in the root directory (do not commit this file):
```env
GROQ_API_KEY=your_groq_api_key_here
```

6. **Run the Streamlit application:**
```bash
streamlit run main.py
```

7. **Use the app:** upload a traffic image, adjust the model/confidence/classes in the sidebar, click **Scan for vehicles**, and review the detection results and AI analysis.

---

## ☁️ Deployment Configuration (Streamlit Cloud)

When deploying to Streamlit Community Cloud:

1. Push your repository without the `.env` file.
2. Make sure `packages.txt` is present at the **repository root** — Streamlit Cloud uses it to `apt-get install` system libraries (`libgl1`, `libglib2.0-0`) that OpenCV needs to import correctly. Without it, deployment fails with `ImportError: libGL.so.1: cannot open shared object file`.
3. In the **Streamlit Cloud Dashboard**, open your application settings:
   * Navigate to **Settings** > **Secrets**.
   * Add your Groq API key:
     ```toml
     GROQ_API_KEY = "your_actual_groq_api_key"
     ```
4. Set the main file path to `main.py`.
5. Deploy the application.

---

## 👤 Author

* **GitHub:** [@your-username](https://github.com/your-username)
