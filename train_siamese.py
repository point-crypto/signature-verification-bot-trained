import os, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from siamese_model import build_siamese

IMG_W, IMG_H = 300, 150
VALID_EXT = (".jpg", ".jpeg", ".png")

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img.astype("float32") / 255.0
    return img.reshape(IMG_H, IMG_W, 1)

def create_pairs(genuine_dir, forged_dir):
    pairs, labels = [], []

    genuine_files = [
        os.path.join(genuine_dir, f)
        for f in os.listdir(genuine_dir)
        if f.lower().endswith(VALID_EXT)
    ]

    forged_files = [
        os.path.join(forged_dir, f)
        for f in os.listdir(forged_dir)
        if f.lower().endswith(VALID_EXT)
    ]

    # ---------- Genuine pairs (label = 1) ----------
    for i in range(len(genuine_files) - 1):
        img1 = load_img(genuine_files[i])
        img2 = load_img(genuine_files[i + 1])

        if img1 is None or img2 is None:
            print(f"⚠ Skipped corrupted genuine image at index {i}")
            continue

        pairs.append([img1, img2])
        labels.append(1)

    # ---------- Forged pairs (label = 0) ----------
    for g in genuine_files:
        img1 = load_img(g)
        if img1 is None:
            continue

        for f in forged_files[:2]:
            img2 = load_img(f)
            if img2 is None:
                print("⚠ Skipped corrupted forged image")
                continue

            pairs.append([img1, img2])
            labels.append(0)

    return np.array(pairs), np.array(labels)

# -------- MAIN --------
pairs, labels = create_pairs(
    "train_data/genuine",
    "train_data/forged"
)

print(f"✅ Total pairs created: {len(pairs)}")

X1 = pairs[:, 0]
X2 = pairs[:, 1]

X1t, X1v, X2t, X2v, yt, yv = train_test_split(
    X1, X2, labels, test_size=0.2, random_state=42
)

model = build_siamese()
model.summary()

model.fit(
    [X1t, X2t],
    yt,
    validation_data=([X1v, X2v], yv),
    epochs=10,
    batch_size=8
)

model.save("model.h5")
print("✅ model.h5 created successfully")
