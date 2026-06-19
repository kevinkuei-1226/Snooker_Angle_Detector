import sys

# Your active environment path
env_path = "/Users/kevinkuei/Desktop/Snooker/Snooker Tool/Cue Line Detector/snooker-env/lib/python3.13/site-packages"
if env_path not in sys.path:
    sys.path.insert(0, env_path)


import cv2
import numpy as np

def get_average_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # 1. Select ROI
    roi = cv2.selectROI("Select ROI", img, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    
    if w > 0 and h > 0:
        cropped = img[y:y+h, x:x+w]
        
        # 2. Convert to HSV
        hsv_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        
        # 3. Calculate mean for each channel
        avg_h = np.mean(hsv_cropped[:, :, 0])
        avg_s = np.mean(hsv_cropped[:, :, 1])
        avg_v = np.mean(hsv_cropped[:, :, 2])
        
        print(f"Average HSV values:")
        print(f"Hue (H): {avg_h:.2f}")
        print(f"Saturation (S): {avg_s:.2f}")
        print(f"Value (V): {avg_v:.2f}")
        
        # Suggesting bounds based on the average
        print(f"\nSuggested lower bound: np.array([{int(avg_h-10)}, {int(avg_s-50)}, {int(avg_v-50)}])")
        print(f"Suggested upper bound: np.array([{int(avg_h+10)}, 255, 255])")

if __name__ == "__main__":
    get_average_color("Data/sticky note 1.jpeg")