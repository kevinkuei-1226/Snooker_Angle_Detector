import cv2
import numpy as np
from cue_engine import CueAngleEngine # <--- This is the link!

def main(video_path, pixel_threshold_range, optimal_window_size, buffer_size=1, angle_precision=0):
    # 1. Initialize the engine
    engine = CueAngleEngine(buffer_size=buffer_size) 
    
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
    
    # taking largest threshold here to filter out the ambiguous bright pixels
    opt_threshold = opt_threshold_stats['end']

    print(f"optimal threshold: {opt_threshold}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 4. Call the engine
        box_angle, line_angle = engine.get_angle(cropped_image=frame[y:y+h, x:x+w],
                                                 pixel_brightness_threshold=opt_threshold)
        
        # 5. Smooth them using the engine's internal memory
        box_angle_smoothed = engine.smooth_angle(box_angle, engine.box_history)
        line_angle_smoothed = engine.smooth_angle(line_angle, engine.line_history)

        # --- NEW SENSOR FUSION FILTERING ---

        if not box_angle or not line_angle:
            final_angle = None
        elif box_angle == 0.00 or box_angle == 90.00:
            # Case A: The box has snapped to the grid lines (blind spot). Rely entirely on the line.
            final_angle = line_angle_smoothed
        elif abs(box_angle - line_angle) > 10.0:
            # Case B: They wildly disagree. When moving near vertical, the box's structural axis 
            # anchor is typically more stable than fitLine's pixel slope regression.
            final_angle = box_angle_smoothed
        else:
            # Case C: They are clean and agree. Average them together to eliminate individual noise!
            final_angle = (box_angle_smoothed + line_angle_smoothed) / 2.0



        print(
            f"Unsmoothed Box Angle: {f'{90 - box_angle:.2f}' if box_angle else 'N/A'}, "
            f"Unsmoothed Line Angle: {f'{abs(90 - line_angle):.2f}' if line_angle else 'N/A'}, "
            f"Final Angle: {f'{90 - final_angle:.2f}' if final_angle is not None else 'N/A'}"
        )
        


        angle_precision = max(0, int(angle_precision))

        cv2.putText(
            frame, 
            f"{f'{(90 - final_angle):.{angle_precision}f}' if final_angle is not None else 'N/A'}",
            (50, 100),                # 1. Bumped Y coordinate down slightly so large text doesn't clip off-screen
            cv2.FONT_HERSHEY_SIMPLEX, # 2. Explicitly named the font type
            2.0,                      # 3. FONT SCALE (Bigger number = Bigger text, default was 1.0 or 2.0)
            (0, 255, 0),              # 4. COLOR (Green)
            4                         # 5. THICKNESS (Higher number = Thicker/Bolder text, default was 1 or 2)
        )
            
        cv2.imshow("Main View", frame)
        key = cv2.waitKey(1) & 0xFF

        # 1. If Spacebar is pressed (ASCII code 32)
        if key == 32:  
            print("Video Paused. Press Spacebar again to resume...")
            while True:
                # Freeze here and wait for another key press
                pause_key = cv2.waitKey(0) & 0xFF  # waitKey(0) blocks indefinitely until a key is pressed
                if pause_key == 32:  # If spacebar is pressed again, break out of the pause loop
                    print("Resuming video.")
                    break
                elif pause_key == ord('q'):  # Allow quitting while paused
                    break

        # 2. If 'q' is pressed normally during playback, break the main loop
        if key == ord('q'): 
            break

def test():
    # testing out certain images and angles with box vs. line logic and accuracy
    a = 1


if __name__ == "__main__":
    video_path = "Data/20260620 EB Birds eye view 1.mp4"
    pixel_threshold_range = list(range(100,256,1))
    main(video_path=video_path,
         pixel_threshold_range=pixel_threshold_range,
         optimal_window_size=30,
         angle_precision=1,
         buffer_size=5)