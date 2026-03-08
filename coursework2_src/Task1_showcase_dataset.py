import os
import cv2
import math
import matplotlib.pyplot as plt

# Root dataset folder
dataset_path = "../dataset"

sequences = ["FD", "HG"]
conditions = ["with_object", "without_object"]

for seq in sequences:
    for cond in conditions:

        folder = os.path.join(dataset_path, seq, cond)
        image_files = sorted(os.listdir(folder))

        images = []
        for img_name in image_files:
            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append((img_name, img))

        n = len(images)
        cols = 3
        rows = math.ceil(n / cols)

        plt.figure(figsize=(12, 4 * rows))
        plt.suptitle(f"{seq} - {cond}", fontsize=16)

        for i, (name, img) in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.title(name, fontsize=8)
            plt.axis("off")

        # plt.tight_layout()
        plt.show()