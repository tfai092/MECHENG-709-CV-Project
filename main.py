import cv2
import os
import numpy as np

# Input/output mapping
sets = [
    ("WeldGapImages/Set 1", "Set1"),
    ("WeldGapImages/Set 2", "Set2"),
    ("WeldGapImages/Set 3", "Set3")
]

for input_dir, set_label in sets:
    output_dir = f"InterimResultsOf{set_label}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_list = sorted(os.listdir(input_dir))

    print(f"Processing {set_label}...")

    for image_name in image_list:
        if not image_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(input_dir, image_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load {image_name}")
            continue

        # --- Step 1: Grayscale ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- Step 2: CLAHE (contrast boost) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # --- Step 3: Blur ---
        blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)

        # --- Step 4: Edge detection (NO threshold before this) ---
        edges = cv2.Canny(blur, 50, 150)

        # --- Step 5: Morphological cleanup ---
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # --- Step 6: Optional ROI (center strip) ---
        height, width = edges.shape
        roi = np.zeros_like(edges)
        roi[:, width//2 - 80: width//2 + 80] = edges[:, width//2 - 80: width//2 + 80]

        # --- Step 7: Hough Line Detection ---
        lines = cv2.HoughLinesP(
            roi,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=100,
            maxLineGap=20
        )

        # --- Step 8: Pick best vertical line ---
        x_position = width // 2  # fallback

        if lines is not None:
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

                if abs(angle) > 80:  # near vertical
                    vertical_lines.append(line[0])

            if len(vertical_lines) > 0:
                # Average x position of vertical lines
                xs = [(l[0] + l[2]) // 2 for l in vertical_lines]
                x_position = int(np.mean(xs))

        # --- Save interim images ---
        base_name = image_name.replace(".jpg", "")

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Gray.jpg"), gray)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Enhanced.jpg"), enhanced)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Edges.jpg"), edges)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_ROI.jpg"), roi)

        # --- Draw result ---
        result_img = img.copy()
        cv2.line(result_img, (x_position, 0), (x_position, height), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_A_WeldGapPosition.jpg"), result_img)

    print(f"{set_label} done.\n")

print("All processing complete.")