import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

dataset_path = r"C:\RDD\Cracks-and-Potholes-in-Road-Images-Dataset-master\Dataset"

TP = FP = FN = TN = 0
image_count = 0
max_images = 25

for folder in os.listdir(dataset_path):

    if image_count >= max_images:
        break

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    raw_img = None
    mask_img = None

    # Load images
    for file in os.listdir(folder_path):

        if "raw" in file.lower():
            raw_img = cv2.imread(os.path.join(folder_path, file))

        elif "pothole" in file.lower():
            mask_img = cv2.imread(os.path.join(folder_path, file), 0)

    if raw_img is None or mask_img is None:
        continue

    image_count += 1

    # Resize
    raw_img = cv2.resize(raw_img, (512, 512))
    mask_img = cv2.resize(mask_img, (512, 512))

    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # IMPROVED DETECTION PIPELINE
    # -------------------------------

    # CLAHE
    clahe = cv2.createCLAHE(2.5, (8,8))
    enhanced = clahe.apply(gray)

    # Bilateral filter
    blur = cv2.bilateralFilter(enhanced, 7, 50, 50)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Edge detection
    edges = cv2.Canny(blur, 60, 150)

    # Combine
    combined = cv2.bitwise_or(edges, thresh)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detection_mask = np.zeros_like(mask_img)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Balanced filtering
        if area < 300:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        if aspect_ratio < 0.2 or aspect_ratio > 4:
            continue

        roi = gray[y:y+h, x:x+w]

        mean_intensity = np.mean(roi)
        std_dev = np.std(roi)

        # Soft filters
        if mean_intensity > 200:
            continue

        if std_dev < 5:
            continue

        # Draw detected region
        cv2.drawContours(detection_mask, [cnt], -1, 255, -1)

    # -------------------------------
    # METRICS CALCULATION
    # -------------------------------

    gt = (mask_img > 0).astype(np.uint8)
    pred = (detection_mask > 0).astype(np.uint8)

    TP += np.sum((pred == 1) & (gt == 1))
    FP += np.sum((pred == 1) & (gt == 0))
    FN += np.sum((pred == 0) & (gt == 1))
    TN += np.sum((pred == 0) & (gt == 0))

# -------------------------------
# FINAL METRICS
# -------------------------------

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

print("\n📊 Evaluation Results")
print("Images processed:", image_count)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Accuracy:", round(accuracy, 4))

# -------------------------------
# GRAPH 1: METRICS
# -------------------------------

metrics = ['Precision', 'Recall', 'Accuracy']
values = [precision, recall, accuracy]

plt.figure()
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("Performance Metrics")

for i, v in enumerate(values):
    plt.text(i, v + 0.02, str(round(v,2)), ha='center')

plt.savefig("metrics_graph.png")
plt.show()

# -------------------------------
# GRAPH 2: CONFUSION MATRIX
# -------------------------------

cm = [[TP, FP],
      [FN, TN]]

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")

plt.xticks([0,1], ['Pred_Pos','Pred_Neg'])
plt.yticks([0,1], ['Actual_Pos','Actual_Neg'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center', va='center')

plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.show()