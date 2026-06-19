import cv2
import numpy as np
import statistics
# import time
# import itertools

class CueAngleEngine:
    def __init__(self, threshold=200):
        self.threshold = threshold


    # ================================================================================================
    # based on a provided image that should contain a white paper surrounding the cue,
    # computes the angle based on the white paper and returns two results:
    # arg 1: the angle estimated by drawing a rectangle on slip (a bit more robust)
    # arg 2: linear regression through all white pixels detected from image
    def get_angle(self,
                  cropped_image, 
                  pixel_brightness_threshold=None,
                ):
    

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
    


    # ================================================================================================
    # finds the optimal threshold by finding the lowest: 
    # (variance within the specified window) + minimal difference between blue line and red box logic,
    # and returns the first threshold value in the window
    def find_optimal_threshold(self, 
                               selected_roi,
                               threshold_range,
                               window_size=20):
        

        # pixel_brightness_threshold_array = list(range(100, 256, 1))

        result_array = []
        for threshold in threshold_range:
            
            #start_time = time.perf_counter()
            red_box_angle, blue_line_angle = self.get_angle(selected_roi, 
                                                                    pixel_brightness_threshold=threshold, 
                                                                    draw_validation_line=False)
            result_array.append((threshold, red_box_angle, blue_line_angle))
            # end_time = time.perf_counter()
            # if red_box_angle is not None and blue_line_angle is not None:
            #     print(f"Threshold: {threshold}, red box angle: {red_box_angle:12.2f}, blue box angle: {blue_line_angle:12.2f}, time taken: {end_time - start_time:.4f} seconds")



        consensus_scores = []
        
        for i in range(len(result_array) - window_size + 1):
            window = result_array[i : i + window_size]
            
            reds = [r[1] for r in window if r[1] is not None]
            blues = [r[2] for r in window if r[2] is not None]
            
            if len(reds) == window_size and len(blues) == window_size:
                m_red, m_blue = statistics.mean(reds), statistics.mean(blues)
                v_red, v_blue = statistics.variance(reds), statistics.variance(blues)
                
                # CONSENSUS LOGIC:
                # 1. We want the difference between red box and blue line to be near zero
                # 2. We want both variances to be low
                diff_score = abs(m_red - m_blue)
                stability_score = v_red + v_blue
                
                # The lower the final score, the higher the consensus
                final_score = diff_score + stability_score
                consensus_scores.append({'start': window[0][0], 'score': final_score, 'mean': (m_red + m_blue) / 2})

        # Sort by the best score and return the top one
        best = min(consensus_scores, key=lambda x: x['score'])
        return best