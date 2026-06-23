import cv2
import numpy as np

# Load your test image
image_path = "Data/sticky note 1.jpeg" 
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    print("--> Select your ROI box using the mouse drag.")
    print("--> Press ENTER or SPACE to confirm the selection.")
    
    # 1. Open the interactive ROI selector window
    roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi
    cv2.destroyWindow("Select ROI")

    if w > 0 and h > 0:
        # Crop the image to your selected region
        roi_cropped = img[y:y+h, x:x+w]
        
        # 2. CONVERT THE CROPPED ROI TO HSV
        roi_hsv = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)
        
        # 3. Calculate mean and standard deviation in HSV space
        mean, std_dev = cv2.meanStdDev(roi_hsv)
        variance = std_dev ** 2
        
        # Extract individual average values
        avg_h = mean[0][0]
        avg_s = mean[1][0]
        avg_v = mean[2][0]
        
        # Print the HSV specific metrics
        # Note: In OpenCV, H is 0-180, S is 0-255, V is 0-255
        print("\n" + "="*30)
        print(f" ROI HSV ANALYSIS ({w}x{h} pixels)")
        print("="*30)
        
        hsv_labels = [
            ('Hue (Color)', avg_h, variance[0][0], std_dev[0][0], 180),
            ('Saturation (Purity)', avg_s, variance[1][0], std_dev[1][0], 255),
            ('Value (Brightness)', avg_v, variance[2][0], std_dev[2][0], 255)
        ]
        
        for label, avg, var, sd, max_val in hsv_labels:
            print(f"{label}:")
            print(f"  - Average: {avg:.2f} / {max_val}")
            print(f"  - Variance: {var:.2f}")
            print(f"  - Std Dev:  {sd:.2f}")
            print("-" * 20)
            
        # 4. GENERATE THE AVERAGE COLOR REFERENCE IMAGE
        # Create a blank 200x200 canvas in HSV format with 8-bit unsigned integers
        avg_color_hsv_patch = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Fill the patch canvas with the rounded average H, S, and V values
        avg_color_hsv_patch[:, :] = [int(round(avg_h)), int(round(avg_s)), int(round(avg_v))]
        
        # Convert it back to standard BGR so your system can save and display it properly
        avg_color_bgr_patch = cv2.cvtColor(avg_color_hsv_patch, cv2.COLOR_HSV2BGR)
        
        # Save the reference color patch to your folder
        output_patch_path = "output/average_color_reference.jpg"
        cv2.imwrite(output_patch_path, avg_color_bgr_patch)
        print(f"Visual color patch saved successfully to: '{output_patch_path}'")
        
        # Optionally show the patch right now on screen
        cv2.imshow("Average Color Reference", avg_color_bgr_patch)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    else:
        print("ROI selection was cancelled or invalid.")