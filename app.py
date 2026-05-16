import cv2
from ultralytics import YOLO

# 1. Custom Trained Safety Helmet Model Loader
model = YOLO("best.pt") 

# 2. GitHub Testing Video Source Link
cap = cv2.VideoCapture("helmet.mp4")

print("Safety Engine Active... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deep learning dynamic prediction layer
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            # Multi-element tracking list array configuration mapping
            coords = box.xyxy.tolist()[0] if hasattr(box.xyxy, 'tolist') else box.xyxy
            if isinstance(coords, list) and isinstance(coords[0], list):
                coords = coords[0]
            x1, y1, x2, y2 = map(int, coords[:4])
            
            conf = float(box.conf)
            cls_id = int(box.cls)
            class_name = model.names[cls_id]

            if conf > 0.25:
                # Custom conditional rules for compliance breach visualization
                if class_name.lower() in ["no-helmet", "without-helmet", "without helmet", "no helmet", "no_helmet"]:
                    color = (0, 0, 255) # Red Box for traffic safety violations
                    label = f"ALERT: No Helmet {conf:.2f}"
                elif class_name.lower() in ["helmet", "with-helmet", "with helmet", "with_helmet"]:
                    color = (0, 255, 0) # Green Box for compliant riders
                    label = f"Safe: Helmet {conf:.2f}"
                else:
                    color = (255, 255, 0) # General safety entities bounding track
                    label = f"{class_name.upper()} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Multi-Class Traffic Compliance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
