import cv2
import numpy as np
from cue_engine import CueAngleEngine # <--- This is the link!

def main(video_path, pixel_threshold_range, optimal_window_size):
    # 1. Initialize the engine
    engine = CueAngleEngine() 
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    # 2. Setup ROI
    roi = cv2.selectROI("Select ROI", frame, False)
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    cropped_first_frame = frame[y:y+h, x:x+w]

    # 3. Calibrate Threshold
    opt_threshold_stats = engine.find_optimal_threshold(selected_roi=cropped_first_frame,
                                                  threshold_range=pixel_threshold_range,
                                                  window_size=optimal_window_size)
    
    opt_threshold = opt_threshold_stats['start']

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 4. Call the engine
        box_angle, line_angle = engine.get_angle(cropped_image=frame[y:y+h, x:x+w],
                                                 pixel_brightness_threshold=opt_threshold)
        
        # box_angle seems more robust, but always open to change
        angle = box_angle

        if angle is not None:
            print(f"Angle: {angle:.2f}")
            cv2.putText(frame, f"{(90 - angle):.2f}", (50,50), 1, 2, (0,255,0))
            
        cv2.imshow("Main View", frame)
        if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    video_path = "Data/Cue with paper between head and hand 1.mp4"
    pixel_threshold_range = list(range(1,256,1))
    main(video_path=video_path,
         pixel_threshold_range=pixel_threshold_range,
         optimal_window_size=30)