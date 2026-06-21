import sys

# Your active environment path
env_path = "/Users/kevinkuei/Desktop/Snooker/Snooker Tool/Cue Line Detector/snooker-env/lib/python3.13/site-packages"
if env_path not in sys.path:
    sys.path.insert(0, env_path)

import cv2
import numpy as np
import statistics
import time
import itertools

# This function analyzes the angle of the white paper to a validation line

def track_paper_angle(cropped_image, 
                           pixel_brightness_threshold=None,
                           draw_validation_line=False
                           ):
    
    # ==========================================
    # STEP 0: IMAGE SETUP AND RESIZING
    # ==========================================


    box_roi = cropped_image.copy()


    # ==========================================
    # STEP 3: COMPUTER VISION TRACKING
    # ==========================================

    # threshold method that looks at brightness, doesn't work as well when hand is in the way
    if pixel_brightness_threshold is not None:

        gray = cv2.cvtColor(box_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Apply a heavy blur to smear the wood grain
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # 2. Strict Threshold: Only the absolute brightest pixels survive
        _, thresh = cv2.threshold(blurred, pixel_brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # 3. Find the outlines (contours)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else: # HSV filtering method, more robust when using a special color
        hsv = cv2.cvtColor(box_roi, cv2.COLOR_BGR2HSV)

        # 2. Split the HSV image into separate channels
        h, s, v = cv2.split(hsv)

        # 3. Apply Median Blur to the Value channel only
        # This smooths out the glare without messing up the color/saturation data
        v_blurred = cv2.medianBlur(v, 5)

        # 4. Merge back together (or just use the v_blurred channel for your mask)
        hsv_blurred = cv2.merge([h, s, v_blurred])

        # 5. Now create your mask using the blurred HSV data
        lower_color_bound = np.array([161, 76, 204]) 
        upper_color_bound = np.array([181, 255, 255]) # Tightened upper bound
        mask = cv2.inRange(hsv_blurred, lower_color_bound, upper_color_bound)
        # 5. Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest white shape by area
        largest_blob = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_blob) > 50:
            
            # --- 1. RED BOX LOGIC ---
            rect = cv2.minAreaRect(largest_blob)
            # box = cv2.boxPoints(rect)
            # box = np.int32(box) 
            # cv2.drawContours(box_roi, [box], 0, (0, 0, 255), 2)
            
            raw_box_angle = rect[2]
            width, height = rect[1]
            if width < height:
                box_angle = abs(raw_box_angle)
            else:
                box_angle = abs(90 - raw_box_angle)

            # --- 2. BLUE LINE LOGIC ---
            [vx, vy, cx, cy] = cv2.fitLine(largest_blob, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, cx, cy = float(vx[0]), float(vy[0]), float(cx[0]), float(cy[0])

            line_angle = np.abs(np.degrees(np.arctan2(vy, vx)))


    # returns red box vertical angle an blue line angle
    if 'box_angle' in locals() and 'line_angle' in locals():
        return abs(90 - box_angle), line_angle
    return None, None

# finds optimal threshold based on threshold logic
def find_optimal_threshold(result_array, window_size=20):
    # This dictionary will store our "consensus" scores
    consensus_scores = []
    
    for i in range(len(result_array) - window_size + 1):
        window = result_array[i : i + window_size]
        
        reds = [r[1] for r in window if r[1] is not None]
        blues = [r[2] for r in window if r[2] is not None]
        
        if len(reds) == window_size and len(blues) == window_size:
            m_red, m_blue = statistics.mean(reds), statistics.mean(blues)
            v_red, v_blue = statistics.variance(reds), statistics.variance(blues)
            
            # CONSENSUS LOGIC:
            # 1. We want the difference between red and blue to be near zero
            # 2. We want both variances to be low
            diff_score = abs(m_red - m_blue)
            stability_score = v_red + v_blue
            
            # The lower the final score, the higher the consensus
            final_score = diff_score + stability_score
            consensus_scores.append({'start': window[0][0], 'score': final_score, 'mean': (m_red + m_blue) / 2})

    # Sort by the best score and return the top one
    best = min(consensus_scores, key=lambda x: x['score'])
    return best


# finds optimal color range based on HSV filtering method
def find_optimal_color_range(cropped_image, center_h, center_s, center_v):
    best_score = float('inf')
    best_bounds = None
    
    # We sweep 'Delta' - how much wiggle room to give the color
    # Hue delta: 5-20, Saturation delta: 20-100, Value delta: 20-100
    for h_d, s_d, v_d in itertools.product(range(5, 21, 5), range(20, 101, 20), range(20, 101, 20)):
        
        lower = np.array([max(0, center_h - h_d), max(0, center_s - s_d), max(0, center_v - v_d)])
        upper = np.array([min(180, center_h + h_d), min(255, center_s + s_d), min(255, center_v + v_d)])
        
        mask = cv2.inRange(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV), lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # Scoring: We want a high area (to see the marker) 
            # and low variance in the resulting fitLine (stability)
            score = 1.0 / (cv2.contourArea(largest) + 0.001) 
            
            if score < best_score:
                best_score = score
                best_bounds = (lower, upper)
                
    return best_bounds



if __name__ == "__main__":
    # Ensure this matches the name of your uploaded test file
    # test_image = "Data/Cue with Paper 4.jpeg" # 99.87 degrees from validation, or 80.13 degrees from validation
    # test_image = "Data/Cue with Paper 5.jpeg" # 86.05 degrees from validation
    # test_image = "Data/Cue with Paper 6.jpeg" # 90.52 degrees from validation, or 89.48 degrees from validation
    # test_image = "Data/Cue with Paper 3.jpeg" # 60.35 degrees from validation, or 119.65 degrees from validation
    test_image = "Data/snapshot_20260618-202104_frame_151.jpg"

    img = cv2.imread(test_image)
    if img is None:
        print(f"Error: Could not load image at {test_image}")

    # Resize massive smartphone photos
    height, width = img.shape[:2]
    
    max_height = 800
    if height > max_height:
        scaling_factor = max_height / float(height)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    cropping_box=(140, 480, 100, 400)
    cropped_img = img[cropping_box[2]:cropping_box[3], cropping_box[0]:cropping_box[1]] if cropping_box else img

    # print(find_optimal_color_range(cropped_image=cropped_img, 
    #                                center_h=172, 
    #                                center_s=126, 
    #                                center_v=254))


    pixel_brightness_threshold_array = list(range(100, 256, 1))

    result_array = []
    for threshold in pixel_brightness_threshold_array:
        start_time = time.perf_counter()

        red_box_angle, blue_line_angle = track_paper_angle(cropped_img, 
                                                                pixel_brightness_threshold=threshold, 
                                                                draw_validation_line=False)
        result_array.append((threshold, red_box_angle, blue_line_angle))
        end_time = time.perf_counter()
        if red_box_angle is not None and blue_line_angle is not None:
            print(f"Threshold: {threshold}, red box angle: {red_box_angle:12.2f}, blue box angle: {blue_line_angle:12.2f}, time taken: {end_time - start_time:.4f} seconds")

          

    # finding the optimal threshold based on the consensus logic

    best_config = find_optimal_threshold(result_array)
    print(f"Optimal threshold start: {best_config['start']}, Best estimate: {best_config['mean']:.2f}")


