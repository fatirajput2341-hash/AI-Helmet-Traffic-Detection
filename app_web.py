import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="AI Traffic Compliance", layout="centered")
st.title("🚦 AI Traffic Safety & Helmet Detection Dashboard")
st.write("Upload a traffic video file below to run compliance checking live on web browser.")

# Loading local trained best weights file
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Running predictions directly on the current frame array
        results = model(frame, conf=0.25)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Direct extraction method compatible with modern YOLOv8 structures
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                # Match against specific deployment class indexes
                if class_name.lower() in ["no-helmet", "without-helmet", "without helmet", "no helmet", "no_helmet"]:
                    color = (0, 0, 255) # Red Box for critical safety breaches
                    label = f"VIOLATION: No Helmet {conf:.2f}"
                elif class_name.lower() in ["helmet", "with-helmet", "with helmet", "with_helmet"]:
                    color = (0, 255, 0) # Green Box for compliant riders
                    label = f"Safe: Helmet {conf:.2f}"
                else:
                    color = (255, 255, 0) # Cyan Box for general traffic targets
                    label = f"{class_name.upper()} {conf:.2f}"

                # Render overlays onto the web pipeline frames
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert to RGB color map format to prevent distorted colors on browser
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        
    cap.release()
