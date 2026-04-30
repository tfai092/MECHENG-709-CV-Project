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

# --- Step 1: Grayscale ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 2: CLAHE ---
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# --- Step 3: Blur ---
blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

# --- Step 4: Sobel X (vertical gradient) ---
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobelx = np.absolute(sobelx)
sobelx = np.uint8(sobelx)

# --- Step 5: Canny ---
edges = cv2.Canny(blur, 70, 150)

# --- Step 6: Morphology (separate stages) ---
kernel = np.ones((3, 3), np.uint8)

edges_open = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
edges_close = cv2.morphologyEx(edges_open, cv2.MORPH_CLOSE, kernel)

# --- Step 7: ROI ---
height, width = edges.shape
roi = np.zeros_like(edges_close)
roi[:, width//2 - 80: width//2 + 80] = edges_close[:, width//2 - 80: width//2 + 80]

# --- Step 8: Hough ---
lines = cv2.HoughLinesP(
    roi,
    1,
    np.pi / 180,
    threshold=50,
    minLineLength=100,
    maxLineGap=20
)

# --- Step 9: Visualise detected lines ---
line_vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

x_position = width // 2

if lines is not None:
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

        # draw ALL detected lines (red)
        cv2.line(line_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if abs(angle) > 80:
            vertical_lines.append(line[0])
            # highlight vertical ones (green)
            cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(vertical_lines) > 0:
        xs = [(l[0] + l[2]) // 2 for l in vertical_lines]
        x_position = int(np.mean(xs))

# --- Final result ---
result_img = img.copy()
cv2.line(result_img, (x_position, 0), (x_position, height), (0, 255, 0), 2)

# --- Save all stages ---
base_name = image_name.replace(".jpg", "")

cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_gray.jpg"), gray)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_enhanced.jpg"), enhanced)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_blur.jpg"), blur)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_sobelx.jpg"), sobelx)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_5_canny.jpg"), edges)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_6_open.jpg"), edges_open)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_7_close.jpg"), edges_close)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_8_roi.jpg"), roi)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_9_lines.jpg"), line_vis)
cv2.imwrite(os.path.join(output_dir, f"{base_name}_10_result.jpg"), result_img)

# --- Optional: show live ---
cv2.imshow("Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Test complete. Check TestOutput folder.")