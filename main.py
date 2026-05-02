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

        # =========================================================
        # 🔁 EXACT pipeline from first code
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

        # =========================================================
        # Everything else EXACTLY as your original second script
        # =========================================================

        # --- Step 7: Hough Line Detection ---
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=50,
            minLineLength=100,
            maxLineGap=20
        )

        # --- Step 8: Pick best vertical line ---
        height, width = edges.shape
        x_position = width // 2  # fallback

        if lines is not None:
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

                if abs(angle) > 80:
                    vertical_lines.append(line[0])

            if len(vertical_lines) > 0:
                xs = [(l[0] + l[2]) // 2 for l in vertical_lines]
                x_position = int(np.mean(xs))

        # --- Save interim images ---
        base_name = image_name.replace(".jpg", "")

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Contrast.jpg"), contrast)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Blackhat.jpg"), blackhat)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Threshold.jpg"), bh_thresh)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_B_Edges.jpg"), edges)

        # --- Draw result ---
        result_img = img.copy()
        cv2.line(result_img, (x_position, 0), (x_position, height), (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_A_WeldGapPosition.jpg"), result_img)

    print(f"{set_label} done.\n")

print("All processing complete.")
