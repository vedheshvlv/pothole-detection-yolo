import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

dataset_path = r"C:\RDD\Cracks-and-Potholes-in-Road-Images-Dataset-master\Dataset"

TP = FP = FN = TN = 0
image_count = 0
max_images = 25

for folder in os.listdir(dataset_path):

    if image_count >= max_images:
        break

    folder_path = os.path.join(dataset_path, folder)

    raw_img = None
    mask_img = None

    for file in os.listdir(folder_path):

        if "raw" in file.lower():
            raw_img = cv2.imread(os.path.join(folder_path, file))

        elif "pothole" in file.lower():
            mask_img = cv2.imread(os.path.join(folder_path, file), 0)

    if raw_img is None or mask_img is None:
        continue

    image_count += 1

    raw_img = cv2.resize(raw_img, (640, 640))
    mask_img = cv2.resize(mask_img, (640, 640))

    results = model(raw_img, conf=0.25)[0]

    pred_mask = np.zeros_like(mask_img)

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(pred_mask, (x1, y1), (x2, y2), 255, -1)

    gt = (mask_img > 0).astype(np.uint8)
    pred = (pred_mask > 0).astype(np.uint8)

    TP += np.sum((pred == 1) & (gt == 1))
    FP += np.sum((pred == 1) & (gt == 0))
    FN += np.sum((pred == 0) & (gt == 1))
    TN += np.sum((pred == 0) & (gt == 0))

# Metrics
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)

print("\n YOLO Evaluation Results")
print("Images processed:", image_count)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Accuracy:", round(accuracy, 4))

# ---------------- GRAPH 1 ----------------
metrics = ['Precision', 'Recall', 'Accuracy']
values = [precision, recall, accuracy]

plt.figure()
plt.bar(metrics, values)
plt.ylim(0, 1)
plt.title("YOLO Performance Metrics")

for i, v in enumerate(values):
    plt.text(i, v + 0.02, str(round(v, 2)), ha='center')

plt.savefig("yolo_metrics.png")
plt.show()

# ---------------- GRAPH 2 ----------------
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
plt.savefig("yolo_confusion_matrix.png")
plt.show()