import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from roboflow import Roboflow
from ultralytics import YOLO
import supervision as sv

st.set_page_config(page_title="Vehicle Type Detection", page_icon="🚗", layout="centered")

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY')
if ROBOFLOW_API_KEY:
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("tanzim-mostafa").project("p2_dhaka_dataset-f6ba6")
        version = project.version(29)
        dataset = version.download("yolov8")
    except Exception as e:
        st.warning(f"Roboflow dataset download failed: {e}")

@st.cache_resource
def load_model():
    YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', 'yolov8x.pt')
    return YOLO(YOLO_WEIGHTS)

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# Map selected class names to IDs
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[name]
    for name in SELECTED_CLASS_NAMES if name in CLASS_NAMES_DICT.values()
]

# Initialize Supervision Annotators
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

# Web App Interface
st.title("🚗 Vehicle Type Detection")
st.write("Upload an image to detect vehicles using YOLOv8 and Supervision.")

uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image into memory and convert to OpenCV BGR format
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image_pil, caption="Original Image", use_container_width=True)

    if st.button("Detect Vehicles"):
        with st.spinner("Analyzing image..."):
            # Run inference
            results = model(image_bgr, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter detections based on selected classes
            if len(SELECTED_CLASS_IDS) > 0:
                detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
            
            # Extract labels and confidences
            labels = [
                f"{model.model.names[class_id]} {confidence:.2f}" 
                for confidence, class_id in zip(detections.confidence, detections.class_id)
            ]
            
            # Apply annotations
            annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # Convert back to RGB for Streamlit rendering
            final_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            st.success("Detection Complete!")
            st.image(final_image, caption="Annotated Image", use_container_width=True)
            
            # Display text results
            if labels:
                st.write("**Detected Objects:**")
                for label in labels:
                    st.write(f"* {label}")
            else:
                st.write("No targeted vehicles detected.")