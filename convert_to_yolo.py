import cv2
import os

dataset_path = r"C:\RDD\Cracks-and-Potholes-in-Road-Images-Dataset-master\Dataset"

output_images = r"C:\Users\vedes\OneDrive\Desktop\pothole_detection_project\dataset\images\train"
output_labels = r"C:\Users\vedes\OneDrive\Desktop\pothole_detection_project\dataset\labels\train"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    raw_img = None
    mask_img = None

    for file in os.listdir(folder_path):
        if "raw" in file.lower():
            raw_img = os.path.join(folder_path, file)

        if "pothole" in file.lower():
            mask_img = os.path.join(folder_path, file)

    if raw_img is None or mask_img is None:
        continue

    image = cv2.imread(raw_img)
    mask = cv2.imread(mask_img, 0)

    h, w = mask.shape

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []

    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)

        x_center = (x + wc/2) / w
        y_center = (y + hc/2) / h
        width = wc / w
        height = hc / h

        label_lines.append(f"0 {x_center} {y_center} {width} {height}")

    name = folder + ".jpg"

    cv2.imwrite(os.path.join(output_images, name), image)

    with open(os.path.join(output_labels, folder + ".txt"), "w") as f:
        f.write("\n".join(label_lines))

print("Conversion done ✅")