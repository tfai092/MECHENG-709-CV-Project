import cv2
import os
import numpy as np

# =========================
# CONFIG
# =========================
# SET TO FALSE BEFORE FINAL SUBMISSION TO OBEY THE BRIEF'S ROI RULE
DEBUG_FULL_IMAGE = False  

sets = ["Set 1", "Set 2", "Set 3"]
script_dir = os.path.dirname(os.path.abspath(__file__))

# =========================
# PARAMETERS
# =========================
y_center = 70
margin = 50
tolerance = 3 # The strict +/- 3 pixel tolerance for marking

# thresholds per set
score_thresholds = {
    "Set 1": 40,
    "Set 2": 50,
    "Set 3": 200
}

# =========================
# PROCESS EACH SET
# =========================
for set_name in sets:

    print(f"\nProcessing {set_name}...")

    input_path = os.path.join(script_dir, set_name)
    interim_dir = f"InterimResultsOf{set_name.replace(' ', '')}"
    interim_path = os.path.join(script_dir, interim_dir)
    os.makedirs(interim_path, exist_ok=True)

    output_csv = f"PositionResultsOf{set_name.replace(' ', '')}.csv"
    csv_path = os.path.join(interim_path, output_csv)

    results = []

    if not os.path.exists(input_path):
        print(f"Warning: Folder '{input_path}' not found. Skipping.")
        continue

    image_list = sorted(os.listdir(input_path))

    for image_name in image_list:
        if not image_name.lower().endswith((".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_path, image_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        height, width = img.shape[:2]

        # =========================
        # ROI (for processing only)
        # =========================
        roi_process = img[y_center-5:y_center+5, :]
        gray = cv2.cvtColor(roi_process, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # =========================
        # SOBEL X
        # =========================
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.mean(sobelx, axis=0)

        # =========================
        # FIND EDGE PAIRS
        # =========================
        best_pair = None
        best_score = 0

        for i in range(margin, width - margin - 10):
            if sobelx[i] < -5: # left edge
                for j in range(i+3, min(i+15, width - margin)):
                    if sobelx[j] > 5: # right edge
                        gap_width = j - i
                        if 4 <= gap_width <= 10:
                            score = abs(sobelx[i]) + abs(sobelx[j])
                            if score > best_score:
                                best_score = score
                                best_pair = (i, j)

        # =========================
        # FINAL DECISION
        # =========================
        threshold = score_thresholds[set_name]

        if best_pair is not None and best_score > threshold:
            x_position = int((best_pair[0] + best_pair[1]) / 2)
            valid = 1
        else:
            x_position = -1
            valid = 0

        # =========================
        # VISUALISATION & DEBUG
        # =========================
        if DEBUG_FULL_IMAGE:
            vis_img = img.copy()
            draw_y_center = y_center
        else:
            # Cropped ROI as per the brief
            vis_y_start = max(0, y_center - 30)
            vis_y_end = min(height, y_center + 30)
            vis_img = img[vis_y_start:vis_y_end, :].copy()
            draw_y_center = y_center - vis_y_start

        # Draw the main processing line (y=70) in blue (thin)
        cv2.line(vis_img, (0, draw_y_center), (width, draw_y_center), (255, 0, 0), 1)

        if valid == 1:
            # 1. Draw detected center (Green line, thin)
            cv2.line(vis_img, (x_position, 0), (x_position, vis_img.shape[0]), (0, 255, 0), 1)
            
            # 2. Draw tolerance boundaries (Red lines, thin)
            cv2.line(vis_img, (x_position - tolerance, 0), (x_position - tolerance, vis_img.shape[0]), (0, 0, 255), 1)
            cv2.line(vis_img, (x_position + tolerance, 0), (x_position + tolerance, vis_img.shape[0]), (0, 0, 255), 1)
            
            # Print debug text right next to the detected gap
            text_x = min(x_position + 10, width - 180) 
            text_y1 = max(draw_y_center - 10, 20)
            text_y2 = min(draw_y_center + 20, vis_img.shape[0] - 10)
            
            cv2.putText(vis_img, f"X:{x_position} S:{best_score:.1f}/{threshold}", (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(vis_img, f"Gap:{best_pair[1]-best_pair[0]}px", (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        else:
            cv2.putText(vis_img, f"INVALID S:{best_score:.1f}/{threshold}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Original style terminal print
        print(f"{set_name} | {image_name} -> x: {x_position}, valid: {valid}, score: {best_score:.1f}")

        # =========================
        # DEBUG: Sobel profile image
        # =========================
        sobel_img = np.zeros((200, width), dtype=np.uint8)
        s = sobelx - np.min(sobelx)
        if np.max(s) != 0:
            s = s / np.max(s) * 199

        for x in range(width):
            y = int(s[x])
            cv2.circle(sobel_img, (x, 199 - y), 1, 255, -1)

        if best_pair is not None:
            cv2.line(sobel_img, (best_pair[0], 0), (best_pair[0], 199), 255, 1)
            cv2.line(sobel_img, (best_pair[1], 0), (best_pair[1], 199), 255, 1)
            cv2.putText(sobel_img, f"Edges: {best_pair}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # =========================
        # SAVE IMAGES (Strict Naming)
        # =========================
        base_name = image_name.rsplit('.', 1)[0]
        base_name = base_name[0].upper() + base_name[1:]
        
        cv2.imwrite(os.path.join(interim_path, f"{base_name}_A_WeldGapPosition.JPG"), vis_img)
        cv2.imwrite(os.path.join(interim_path, f"{base_name}_B_InterimResult1.jpg"), gray)
        cv2.imwrite(os.path.join(interim_path, f"{base_name}_B_InterimResult2.jpg"), blur)
        cv2.imwrite(os.path.join(interim_path, f"{base_name}_B_InterimResult3.jpg"), sobel_img)
        
        # =========================
        # STORE RESULT
        # =========================
        results.append(f"{image_name}, {x_position}, {valid}")

    # =========================
    # SAVE CSV 
    # =========================
    with open(csv_path, "w") as f:
        f.write('ImageName,Weld gap position in pixel/integer,"Weld gap position valid? 0=false, 1=true"\n')
        for line in results:
            f.write(line + "\n")
