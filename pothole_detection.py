import cv2
import numpy as np
import os

image_folder = r"C:\Users\vedes\OneDrive\Desktop\pothole_detection_project\dataset\images"

output_folder = r"C:\Users\vedes\OneDrive\Desktop\pothole_detection_project\output\final"
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(image_folder):

    path = os.path.join(image_folder, img_name)
    image = cv2.imread(path)

    if image is None:
        continue

    original = image.copy()

    # -------------------------------
    # 1. Grayscale
    # -------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # 2. CLAHE (contrast improvement)
    # -------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # -------------------------------
    # 3. Bilateral Filter (better than Gaussian)
    # -------------------------------
    blur = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # -------------------------------
    # 4. Adaptive Threshold (handles lighting)
    # -------------------------------
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # -------------------------------
    # 5. Edge Detection
    # -------------------------------
    edges = cv2.Canny(blur, 70, 150)

    # -------------------------------
    # 6. Combine both
    # -------------------------------
    combined = cv2.bitwise_or(edges, thresh)

    # -------------------------------
    # 7. Morphology
    # -------------------------------
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    # -------------------------------
    # 8. Find contours
    # -------------------------------
    contours,_ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # 🔥 Area filtering (important)
        if area < 800 or area > 20000:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        # 🔥 Remove long lines
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue

        roi = gray[y:y+h, x:x+w]

        mean_intensity = np.mean(roi)
        std_dev = np.std(roi)

        # 🔥 KEY FILTERS
        if mean_intensity > 180:
            continue

        if std_dev < 12:
            continue

        # -------------------------------
        # 9. Severity classification
        # -------------------------------
        if area < 3000:
            label = "LOW"
            color = (0,255,0)

        elif area < 8000:
            label = "MEDIUM"
            color = (0,165,255)

        else:
            label = "HIGH"
            color = (0,0,255)

        # Draw box
        cv2.rectangle(original,(x,y),(x+w,y+h),color,2)
        cv2.putText(original,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # Save result
    cv2.imwrite(os.path.join(output_folder, img_name), original)

print("Detection Completed ")