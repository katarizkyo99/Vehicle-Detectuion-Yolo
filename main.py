import io
import os

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from PIL import Image
from ultralytics import YOLO

load_dotenv()

st.set_page_config(page_title="TrafficLens", page_icon="🚦", layout="centered")

GROQ_MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]
CLASS_ICONS = {"car": "🚗", "motorcycle": "🏍️", "bus": "🚌", "truck": "🚚"}
CLASS_COLORS = {"car": "#f97316", "motorcycle": "#facc15", "bus": "#ef4444", "truck": "#a855f7"}


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    return YOLO(weights_path)


def image_to_download_bytes(image_rgb: np.ndarray) -> bytes:
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_ai_analysis(class_counts: dict) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is missing — set it in your environment or .env file to enable AI analysis."

    llm = ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL_ID, temperature=0.4)

    if sum(class_counts.values()) == 0:
        summary = "No vehicles were detected in this image."
    else:
        summary = ", ".join(f"{count} {label}(s)" for label, count in class_counts.items() if count > 0)

    prompt = f"""You are a traffic analysis assistant. Based on the following vehicle detection results from a single image, write a short, natural 2-3 sentence analysis of the traffic composition. Mention what appears to dominate, and any notable observation. Do not invent details that aren't implied by the counts.

Detection results: {summary}
Total vehicles detected: {sum(class_counts.values())}

Analysis:"""

    response = llm.invoke(prompt)
    return response.content


# ---- Global styling ----
st.markdown(
    """
    <style>
    .tl-hero {
        background: linear-gradient(135deg, #1e1b2e 0%, #7c2d12 60%, #c2410c 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .tl-hero h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .tl-hero p {
        color: #fdba74;
        font-size: 1rem;
        margin: 0;
    }
    .tl-card {
        background: #18181b;
        border: 1px solid #27272a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .tl-card-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .tl-card-label {
        color: #d4d4d8;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .tl-card-count {
        color: white;
        font-size: 1.4rem;
        font-weight: 800;
    }
    .tl-bar-bg {
        background: #27272a;
        border-radius: 6px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .tl-bar-fill {
        height: 100%;
        border-radius: 6px;
    }
    .tl-total {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #c2410c, #f97316);
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .tl-total .num { font-size: 2.5rem; font-weight: 800; }
    .tl-total .label { font-size: 0.9rem; opacity: 0.9; }
    </style>

    <div class="tl-hero">
        <h1>🚦 TrafficLens</h1>
        <p>Point your camera at the road. Let AI count what's driving through.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")

    yolo_weights = st.selectbox(
        "YOLO model size",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8x.pt"],
        index=0,
        help="Nano/Small are lighter and faster to deploy. X is most accurate but heaviest.",
    )

    confidence = st.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

    try:
        _preview_model = load_model(yolo_weights)
        all_model_classes = sorted(_preview_model.model.names.values())
    except Exception:
        all_model_classes = DEFAULT_CLASSES

    selected_classes = st.multiselect(
        "Classes to detect",
        options=all_model_classes,
        default=[c for c in DEFAULT_CLASSES if c in all_model_classes],
    )

    st.divider()
    enable_analysis = st.toggle("🧠 Generate AI traffic analysis", value=True)
    st.caption(f"Analysis model: `{GROQ_MODEL_ID}`")

# ---- Model load ----
try:
    model = load_model(yolo_weights)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

CLASS_NAMES_DICT = model.model.names
NAME_TO_ID = {value: key for key, value in CLASS_NAMES_DICT.items()}
SELECTED_CLASS_IDS = [NAME_TO_ID[name] for name in selected_classes if name in NAME_TO_ID]

box_annotator = sv.BoxAnnotator(thickness=3)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.8, text_color=sv.Color.BLACK)

uploaded_file = st.file_uploader("Upload a road/traffic image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    if st.button("🔍 Scan for vehicles", use_container_width=True):
        with st.spinner("Scanning..."):
            results = model(image_bgr, conf=confidence, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            if SELECTED_CLASS_IDS:
                detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

            labels = [
                f"{model.model.names[class_id]} {conf_score:.2f}"
                for conf_score, class_id in zip(detections.confidence, detections.class_id)
            ]

            annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            final_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            class_counts = {}
            for class_id in detections.class_id:
                name = model.model.names[class_id]
                class_counts[name] = class_counts.get(name, 0) + 1

        # ---- Tabs instead of side-by-side columns ----
        tab1, tab2 = st.tabs(["📷 Original", "🎯 Detected"])
        with tab1:
            st.image(image_pil, use_container_width=True)
        with tab2:
            st.image(final_image_rgb, use_container_width=True)
            st.download_button(
                "⬇️ Download annotated image",
                data=image_to_download_bytes(final_image_rgb),
                file_name="trafficlens_detected.png",
                mime="image/png",
                use_container_width=True,
            )

        # ---- Total counter banner ----
        total = sum(class_counts.values())
        st.markdown(
            f"""
            <div class="tl-total">
                <div class="num">{total}</div>
                <div class="label">vehicle(s) detected</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if total > 0:
            st.markdown("##### Composition")
            for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                pct = (count / total) * 100
                color = CLASS_COLORS.get(label, "#f97316")
                icon = CLASS_ICONS.get(label, "🚘")
                st.markdown(
                    f"""
                    <div class="tl-card">
                        <div class="tl-card-row">
                            <span class="tl-card-label">{icon} {label.capitalize()}</span>
                            <span class="tl-card-count">{count}</span>
                        </div>
                        <div class="tl-bar-bg">
                            <div class="tl-bar-fill" style="width:{pct:.0f}%; background:{color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No vehicles detected. Try lowering the confidence threshold or selecting more classes.")

        if enable_analysis:
            st.markdown("##### 🧠 AI Analysis")
            with st.spinner("Generating analysis..."):
                analysis = generate_ai_analysis(class_counts)
            st.markdown(analysis)
