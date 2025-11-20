import os
import numpy as np
import joblib
from PIL import Image

import torch
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

# ===========================
# إعداد الجهاز والنموذج
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = EfficientNet_B2_Weights.DEFAULT
model = efficientnet_b2(weights=weights).to(device)
model.eval()
transform = weights.transforms()

# ===========================
# استخراج الخصائص من صورة/patch
# ===========================
def extract_features_from_image(img: Image.Image):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.features(x)
        features = torch.flatten(features, 1).cpu().numpy()
    return features.flatten()

def extract_features(image_path):
    if not os.path.exists(image_path):
        print(f"[!] Image not found: {image_path}")
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[!] Error opening image {image_path}: {e}")
        return None
    return extract_features_from_image(img)

# ===========================
# تقسيم الصورة إلى patches
# ===========================
def split_into_patches(img: Image.Image, patch_size=(224,224), stride=224):
    patches = []
    w, h = img.size
    for y in range(0, h - patch_size[1] + 1, stride):
        for x in range(0, w - patch_size[0] + 1, stride):
            patch = img.crop((x, y, x + patch_size[0], y + patch_size[1]))
            patches.append(patch)
    return patches

# ===========================
# التحقق من صورة باستخدام Hybrid SVM
# ===========================
def is_oct_image(image_path, model_path="oct_svm.joblib",
                        patch_size=(224,224), stride=224,
                        min_oct_patches=1):
    if not os.path.exists(image_path):
        print(f"[!] Image not found: {image_path}")
        return False
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[!] Error opening image {image_path}: {e}")
        return False

    data = joblib.load(model_path)
    model_svm = data["svm"]
    threshold = data["meta"]["threshold"]

    patches = split_into_patches(img, patch_size, stride)

    oct_patches_count = 0
    for patch in patches:
        f = extract_features_from_image(patch)
        score = model_svm.decision_function([f])[0]
        if score > threshold:
            oct_patches_count += 1
        if oct_patches_count >= min_oct_patches:
            return True

    return False

# ===========================
# تدريب One-Class SVM مع جمع الخصائص
# نسخة سريعة للتجربة
# ===========================
def train_oct_svm(oct_folder="D:\\DeepSight\\Graduation-project-\\RetinalOCT_Dataset\\train",
                  save_path="oct_svm.joblib",
                  patch_size=(224,224), stride=224,
                  max_images=20, max_patches=200,
                  svm_nu=0.05,
                  nonoct_folder="D:\\DeepSight\\Graduation-project-\\Non_OCT"):
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import roc_curve

    # --------- جمع خصائص OCT ---------
    X_pos = []
    count_img = 0
    for root, _, files in os.walk(oct_folder):
        for fname in files:
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                path = os.path.join(root, fname)
                try:
                    img = Image.open(path).convert("RGB")
                except:
                    continue
                patches = split_into_patches(img, patch_size, stride)
                for p in patches:
                    X_pos.append(extract_features_from_image(p))
                    if len(X_pos) >= max_patches:
                        break
                count_img += 1
                if count_img >= max_images or len(X_pos) >= max_patches:
                    break
    X_pos = np.array(X_pos)
    print(f"[+] Collected {len(X_pos)} OCT patch features")

    # --------- تدريب SVM ---------
    model_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=svm_nu)
    model_svm.fit(X_pos)
    print("[+] SVM trained on OCT patches")

    # --------- جمع خصائص Non-OCT ---------
    X_neg = []
    count_img = 0
    for root, _, files in os.walk(nonoct_folder):
        for fname in files:
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                path = os.path.join(root, fname)
                try:
                    img = Image.open(path).convert("RGB")
                except:
                    continue
                patches = split_into_patches(img, patch_size, stride)
                for p in patches:
                    X_neg.append(extract_features_from_image(p))
                    if len(X_neg) >= max_patches:
                        break
                count_img += 1
                if count_img >= max_images or len(X_neg) >= max_patches:
                    break
    X_neg = np.array(X_neg)
    print(f"[+] Collected {len(X_neg)} Non-OCT patch features")

    # --------- Calibration threshold ---------
    pos_scores = model_svm.decision_function(X_pos)
    neg_scores = model_svm.decision_function(X_neg)
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    scores = np.concatenate([pos_scores, neg_scores])
    fpr, tpr, thresholds = roc_curve(y, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    print(f"[+] Best threshold set to {best_threshold:.4f}")

    # --------- حفظ النموذج + threshold ---------
    joblib.dump({"svm": model_svm, "meta": {"threshold": float(best_threshold),
                                            "patch_size": patch_size,
                                            "stride": stride,
                                            "svm_nu": svm_nu}}, save_path)
    print(f"[+] Hybrid OCT SVM saved to {save_path}")