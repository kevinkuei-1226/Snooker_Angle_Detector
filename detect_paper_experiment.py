import cv2
import numpy as np
import statistics



# This function analyzes the angle of the white paper to a validation line

def track_white_paper_blob(image_path, 
                           pixel_brightness_threshold,
                           cropping_box=None,
                           draw_validation_line=False
                           ):
    
    # ==========================================
    # STEP 0: IMAGE SETUP AND RESIZING
    # ==========================================

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Resize massive smartphone photos
    height, width = img.shape[:2]
    
    max_height = 800
    if height > max_height:
        scaling_factor = max_height / float(height)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    print("Resized Image Size: {}x{}".format(img.shape[1], img.shape[0]))
    if cropping_box:
        display_img = img[cropping_box[2]:cropping_box[3], cropping_box[0]:cropping_box[1]]
    else:
        display_img = img

        # ==========================================
        # STEP 1: SELECT REGION OF INTEREST (ROI) if no cropping box provided
        # ==========================================
        print("\n" + "="*50)
        print("STEP 1: ROI SELECTION")
        print("Draw a box around the cue. Press SPACE or ENTER to confirm.")
        print("="*50)

        roi = cv2.selectROI("Select Scan Area", display_img, showCrosshair=False)
        cv2.destroyWindow("Select Scan Area")

        if roi[2] == 0 or roi[3] == 0:
            return

        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        display_img = display_img[y:y+h, x:x+w]
    
    box_roi = display_img.copy()
    # ==========================================
    # STEP 2: MANUAL VALIDATION LINE IF DESIRED
    # ==========================================

    # capture mouse clicks
    def capture_clicks(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(manual_points) < 2:
                manual_points.append((mx, my))
                # Draw a small green dot where you click
                cv2.circle(display_img, (mx, my), 4, (0, 255, 0), -1) 
                
                # Draw the line once the second point is clicked
                if len(manual_points) == 2:
                    cv2.line(display_img, manual_points[0], manual_points[1], (0, 255, 0), 2)
                
                cv2.imshow("Click 2 Points", display_img)
    

    manual_angle = None
    if draw_validation_line:
        print("\n" + "="*50)
        print("STEP 2: MANUAL VALIDATION")
        print("Click TWO points along the edge of the cue to draw a green validation line.")
        print("Press any key to calculate results after drawing the line.")
        print("="*50 + "\n")

        manual_points = []
    
        cv2.imshow("Click 2 Points", display_img)
        cv2.setMouseCallback("Click 2 Points", capture_clicks)
        cv2.waitKey(0) # Wait for you to press a key to continue
        cv2.destroyWindow("Click 2 Points")

        # Calculate the angle of your manual line
        if len(manual_points) == 2:
            dx = float(manual_points[1][0] - manual_points[0][0])
            dy = float(manual_points[1][1] - manual_points[0][1])
            manual_angle = np.abs(np.degrees(np.arctan2(dy, dx)))

    # ==========================================
    # STEP 3: COMPUTER VISION TRACKING (HSV)
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
        lower_color_bound = np.array([161, 110, 0]) 
        upper_color_bound = np.array([181, 130, 255]) # Tightened upper bound
        mask = cv2.inRange(hsv_blurred, lower_color_bound, upper_color_bound)

        cv2.imshow("masked with hsv", mask)
        cv2.waitKey(0)
        # 5. Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("=== FINAL DIAGNOSTICS ===")
    if manual_angle is not None:
        print(f"User Validation Line Angle = {manual_angle:.2f} degrees")
        print("-" * 30)

    if contours:
        # Find the largest white shape by area
        print("contour found")
        largest_blob = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_blob) > 20:
            
            # --- 1. RED BOX LOGIC ---
            rect = cv2.minAreaRect(largest_blob)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 
            cv2.drawContours(box_roi, [box], 0, (0, 0, 255), 2)
            
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

            if vy != 0:
                t_top = (0 - cy) / vy
                x_top = int(cx + t_top * vx)
                t_bottom = (box_roi.shape[0] - cy) / vy
                x_bottom = int(cx + t_bottom * vx)
                cv2.line(box_roi, (x_top, 0), (x_bottom, box_roi.shape[0]), (255, 0, 0), 2)

            # --- 3. PRINT RESULTS ---
            print(f"Threshold Used              = {pixel_brightness_threshold}")
            print(f"Red Box Horizontal Angle   = {box_angle:.2f} degrees")
            print(f"Red Box Vertical Angle     = {90 - box_angle:.2f} degrees")
            print(f"Blue Line Angle            = {line_angle:.2f} degrees")
            
            if manual_angle is not None:
                print("-" * 30)
                print(f"Diff: Validation vs Red Box   = {abs(box_angle - abs(90 - manual_angle)):.2f} degrees")
                print(f"Diff: Validation vs Blue Line = {abs(abs(90 - line_angle) - abs(90 - manual_angle)):.2f} degrees")
                
        else:
            print("No paper marker large enough was found.")
    print("=========================\n")

    # Update the final image with the yellow box
    # display_img[y:y+h, x:x+w] = box_roi
    display_img = box_roi.copy()
    # cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Final Tracking vs Validation", display_img)
    # cv2.imshow("Grayed version", gray)
    # cv2.imshow("Blurred version", blurred)
    # cv2.imshow("Pixels survived after threshold", thresh)
    
    # don't wait key when doing validation, uncomment when wanting to view result before destroying windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # returns red box vertical angle an blue line angle
    if 'box_angle' in locals() and 'line_angle' in locals():
        return abs(90 - box_angle), line_angle
    return None, None


if __name__ == "__main__":
    # Ensure this matches the name of your uploaded test file
    # test_image = "Data/Cue with Paper 3.jpeg" # 99.87 degrees from validation, or or 80.13 degrees from validation
    # test_image = "Data/Cue with Paper 5.jpeg" # 86.05 degrees from validation

    # red_box_angle, blue_line_angle = track_white_paper_blob(image_path="output/frame_11.png",
    #                                                         pixel_brightness_threshold=164, 
    #                                                         cropping_box=None,
    #                                                         draw_validation_line=False)

    track_white_paper_blob(image_path="Data/sticky note 2.jpeg",
                           pixel_brightness_threshold=None, 
                           cropping_box=None,
                           draw_validation_line=True)

