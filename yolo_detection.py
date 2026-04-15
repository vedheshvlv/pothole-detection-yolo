import cv2
import os
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

image_folder = r"C:\Users\vedes\OneDrive\Desktop\pothole_detection_project\dataset\images\train"
output_folder = r"C:\Users\vedes\OneDrive\Desktop\pothole_detection_project\output\yolo"

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(image_folder):

    path = os.path.join(image_folder, img_name)
    image = cv2.imread(path)

    if image is None:
        continue

    # 🔥 VERY IMPORTANT FIX: lower confidence
    results = model(image, conf=0.05)[0]

    print(f"{img_name} → Detections:", len(results.boxes))

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Area
        area = (x2 - x1) * (y2 - y1)

        # 🔥 Improved severity classification
        if area < 3000:
            label = "LOW"
            color = (0, 255, 0)

        elif area < 10000:
            label = "MEDIUM"
            color = (0, 165, 255)

        else:
            label = "HIGH"
            color = (0, 0, 255)

        text = f"{label} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(os.path.join(output_folder, img_name), image)

print("YOLO Detection Completed ✅")