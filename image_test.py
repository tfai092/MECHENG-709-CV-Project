import cv2
import os
import numpy as np

# --- Input ---
input_dir = "WeldGapImages/Set 1"
output_dir = "TestOutput"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Get first image ---
image_list = sorted(os.listdir(input_dir))
image_name = next((n for n in image_list if n.lower().endswith(".jpg")), None)

if image_name is None:
    print("No JPG images found.")
    exit()

print(f"Testing on: {image_name}")

img_path = os.path.join(input_dir, image_name)
img = cv2.imread(img_path)

if img is None:
    print("Failed to load image.")
    exit()

# =========================================================
# PREPROCESSING
# =========================================================

# --- Step 1: Grayscale ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 2: Contrast ---
contrast = cv2.convertScaleAbs(gray, alpha=1.7, beta=0)

# --- Step 3: Blur ---
blur = cv2.GaussianBlur(contrast, (5, 5), 0.7)

# --- Step 4: Black-hat ---
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 35))
blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, vertical_kernel)

# --- Step 5: Threshold ---
_, bh_thresh = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

# --- Step 6: Edge detection ---
edges = cv2.Canny(bh_thresh, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    1,
    np.pi / 180,
    threshold=40,
    minLineLength=120,
    maxLineGap=30
)

# =========================================================
# SELECT BEST LINE (vertical + longest)
# =========================================================

result_img = img.copy()
line_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

best_x = None
best_len = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        length = np.hypot(x2 - x1, y2 - y1)

        # draw all lines (red)
        cv2.line(line_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if abs(angle) > 80:  # near vertical
            # highlight vertical (green)
            cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if length > best_len:
                best_len = length
                best_x = (x1 + x2) // 2

# --- Draw final result ---
if best_x is not None:
    cv2.line(result_img, (best_x, 0), (best_x, img.shape[0]), (0, 255, 0), 2)
else:
    print("No weld gap detected.")

# =========================================================
# SAVE OUTPUTS
# =========================================================

base_name = image_name.replace(".jpg", "")

cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_gray.jpg"), gray)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_contrast.jpg"), contrast)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_blur.jpg"), blur)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_blackhat.jpg"), blackhat)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_5_thresh.jpg"), bh_thresh)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_6_edges.jpg"), edges)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_7_lines.jpg"), line_vis)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_8_result.jpg"), result_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Test complete. Check TestOutput folder.")
