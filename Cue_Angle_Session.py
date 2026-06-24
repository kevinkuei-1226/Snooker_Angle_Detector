import cv2
import numpy as np
import statistics
from collections import deque
import math
import Image_Util as Im_Util
# import time
# import itertools

class Cue_Angle_Session:
    def __init__(self, 
                 method="grayScale",
                 threshold=200, 
                 buffer_size=5,
                 Gauss_Blur=[15,15],
                 HSV_lower=None,
                 HSV_upper=None,
                 medianBlur=5):
        self.threshold = threshold
        self.buffer_size = buffer_size
        self.Gauss_Blur = Gauss_Blur
        self.HSV_lower = HSV_lower
        self.HSV_upper = HSV_upper
        self.method = method
        
        if method == "grayScale":
            self.params = [threshold, Gauss_Blur]
        elif method == "HSV":
            self.params = [HSV_lower, HSV_upper,medianBlur]
        else:
            self.params = [threshold, Gauss_Blur]
            print("no valid method provided, will default to grayScale method")

        # Create history & buffers for both angles (stores the last 'buffer_size' frames)
        self.box_history = []
        self.line_history = []
        self.box_buffer = deque(maxlen=buffer_size)
        self.line_buffer = deque(maxlen=buffer_size)


    # ================================================================================================
    # finds the center position of the white paper in the given image
    # returns the position in image coordinates

    def get_position(self,
                     image,
                     show_results=False
                     ):
        box_roi = image.copy()
        
        # 3. Find the outlines (contours)
        # currently just defaulting to grayScale version, can change later
        contours = Im_Util.get_Contours(roi=box_roi,
                                           method=self.method,
                                           params=self.params)

        center_x, center_y = None, None
        
        if contours:
            # 1. Find your largest contour as usual
            largest_blob = max(contours, key=cv2.contourArea)

            # 2. Calculate the spatial moments of the largest contour
            M = cv2.moments(largest_blob)

            # 3. Prevent a division-by-zero error (in case the blob has an area of 0 pixels)
            if M["m00"] != 0:
                # Calculate the X and Y coordinates of the center of mass
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                
                # print(f"Center Mass Position: X={center_x}, Y={center_y}")
                
                # Show annotated image
                if show_results:
                    cv2.circle(box_roi, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.imshow("Position display (press any key to close)", box_roi)
                    cv2.waitKey(0)
        
        return center_x, center_y

    # ================================================================================================
    # based on a provided image that should contain a white paper surrounding the cue,
    # computes the angle based on the white paper and returns two results:
    # arg 1: the angle estimated by drawing a rectangle on slip (a bit more robust)
    # arg 2: linear regression through all white pixels detected from image
    def get_angle(self,
                  image,
                  show_results=False
                ):
    

        box_roi = image.copy()


        # get contours

        contours = Im_Util.get_Contours(roi=box_roi,
                                        method=self.method,
                                        params=self.params)
  
        
        if contours:
            # Find the largest white shape by area
            largest_blob = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_blob) > 30:
                
                # --- 1. RED BOX LOGIC ---
                rect = cv2.minAreaRect(largest_blob)
                
                raw_box_angle = rect[2]
                width, height = rect[1]
                if width < height:
                    box_angle = raw_box_angle
                else:
                    box_angle = 90 - raw_box_angle

                # --- 2. BLUE LINE LOGIC ---
                [vx, vy, cx, cy] = cv2.fitLine(largest_blob, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy, cx, cy = float(vx[0]), float(vy[0]), float(cx[0]), float(cy[0])


                # Force the vector direction to always point 'up' relative to the image screen (y decreases up)
                if vy > 0:
                    vx = -vx
                    vy = -vy
            
                # Calculate the actual angle relative to the horizontal plane
                line_angle = np.degrees(np.arctan2(vy, vx))

                # Shift it so pointing right is 0°, straight up is 90°, and pointing left is 180°
                if line_angle < 0:
                    line_angle += 180
                
                if show_results:
                    # draw blue line

                    if vy != 0:
                        
                        t_top = (0 - cy) / vy
                        x_top = int(cx + t_top * vx)
                        t_bottom = (box_roi.shape[0] - cy) / vy
                        x_bottom = int(cx + t_bottom * vx)
                        cv2.line(box_roi, (x_top, 0), (x_bottom, box_roi.shape[0]), (255, 0, 0), 2)
                    
                    # draw red box
                    box = cv2.boxPoints(rect)
                    box = np.int32(box) 
                    cv2.drawContours(box_roi, [box], 0, (0, 0, 255), 2)
                    display_img = box_roi.copy()

                    # show annotated image
                    cv2.imshow("Red & Blue Tracking", display_img)
                    cv2.waitKey(0)


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
    

        result_array = []
        for threshold in threshold_range:

            red_box_angle, blue_line_angle = self.get_angle(selected_roi, 
                                                            pixel_brightness_threshold=threshold, 
                                                            )
            result_array.append((threshold, red_box_angle, blue_line_angle))

        consensus_scores = []
        
        for i in range(len(result_array) - window_size + 1):
            window = result_array[i : i + window_size]
            
            reds = [r[1] for r in window if r[1] is not None]
            blues = [r[2] for r in window if r[2] is not None]
            
            if len(reds) == window_size and len(blues) == window_size:
                m_red, m_blue = statistics.mean(reds), statistics.mean(blues)
                v_red, v_blue = statistics.variance(reds), statistics.variance(blues)
                
                # CONSENSUS LOGIC:
                # 1. We want the difference between red box and blue line angle to be near zero
                # 2. We want both variances to be low
                # 3. use standard deviation for unit consistency (in degrees)
                # 4. a bit more important for methods to agree than variance to be minimal
                diff_score = abs(m_red - m_blue)
                stability_score = math.sqrt(v_red) + math.sqrt(v_blue)
                
                diff_weight = 0.7
                sd_weight = 0.3

                final_score =diff_weight * diff_score + sd_weight * stability_score
                consensus_scores.append({'start': window[0][0], 'score': final_score, 'mean': (m_red + m_blue) / 2})

        # Sort by the best score and return the top one
        best = min(consensus_scores, key=lambda x: x['score'])
        return best
    

    # smooth out angles in video frames for less jittering
    # buffer size when initializing determines how much smoothing is done
    def get_smoothed_angle(self):
        
        smoothed_box_angle = None
        smoothed_line_angle = None
        if len(self.box_buffer) >= self.buffer_size and len(self.line_buffer) >= self.buffer_size:
            smoothed_box_angle = np.mean(self.box_buffer)
            smoothed_line_angle = np.mean(self.line_buffer)
        

        return smoothed_box_angle, smoothed_line_angle


    # update angle history
    def update_history(self, 
                       new_box_angle, 
                       new_line_angle):
        # Append to rolling buffers for live smoothing
        self.box_buffer.append(new_box_angle)
        self.line_buffer.append(new_line_angle)
        
        # Append to master session logs
        self.box_history.append(new_box_angle)
        self.line_history.append(new_line_angle)
    

if __name__ == "__main__":
    test_image = cv2.imread("output/frame_11_cropped.png")
    CA = Cue_Angle_Session(method="grayScale", 
                        threshold=179)

    print(CA.get_position(image=test_image, show_results=False))
    print(CA.get_angle(image=test_image, show_results=True))
