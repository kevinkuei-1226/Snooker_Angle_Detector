import sys

# Your active environment path
env_path = "/Users/kevinkuei/Desktop/Snooker/Snooker Tool/Cue Line Detector/snooker-env/lib/python3.13/site-packages"
if env_path not in sys.path:
    sys.path.insert(0, env_path)

import cv2
import time
import paper_detection as pd

def run_production(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return
    

    # 1. Manual Crop (One time)
    roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=False)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    
    # 2. Optimization (First frame only)
    print("Calibrating threshold...")
    cropped_first = first_frame[y:y+h, x:x+w]
    result_array = []
    for t in range(100, 256):
        r, b = pd.track_paper_angle(cropped_first, t)
        result_array.append((t, r, b))
    
    best_config = pd.find_optimal_threshold(result_array, window_size=5)
    optimal_t = int(best_config['start'] + 2) # some discretion here on choose which threshold within the window
    print(f"Optimal Threshold found: {optimal_t}")

    # 3. Production Loop
    frame_count = 0
    total_processing_time = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    print(f"{'Frame':<10} | {'Angle (Deg)':<12} | {'Time (s)':<10}")
    print("-" * 40)

    paused = False

    while cap.isOpened():
        if not paused:
            start = time.perf_counter()
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            cropped = frame[y:y+h, x:x+w]
            red, blue = pd.track_paper_angle(cropped, optimal_t)
            end = time.perf_counter()
            total_processing_time += (end - start)

        # Draw the angle on the frame
        if red is not None:
            text = f"Angle: {red:.0f} deg"
            # cv2.putText(image, text, position, font, fontScale, color, thickness)
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw a rectangle showing your ROI (Optional, for visual confirmation)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Live Analysis", frame)

        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Quit
            break
        elif key == ord('k'): # Pause
            paused = not paused
        elif key == ord('s'): # Snapshot
            # Save the raw frame with a timestamp to the 'Data' folder
            timestamp_str = time.strftime("%Y%m%d-%H%M%S")
            filename = f"Data/snapshot_{timestamp_str}_frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved: {filename}")

    cap.release()
    print(f"\nTotal processing time: {total_processing_time:.4f}s")
    print(f"Average time per frame: {total_processing_time/frame_count:.4f}s")

if __name__ == "__main__":
    run_production("Data/Cue with paper between head and hand 1.mp4")