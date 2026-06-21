import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

    # taking mid threshold here
    opt_threshold = opt_threshold_stats['start'] + int(optimal_window_size / 2)

    print(f"optimal threshold: {opt_threshold}")

    angle_history_log = []

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

        if final_angle is not None:
            angle_history_log.append(90 - final_angle)
        else:
            angle_history_log.append(float('nan'))

        print(
            f"Unsmoothed Box Angle: {f'{90 - box_angle:.2f}' if box_angle else 'N/A'}, "
            f"Unsmoothed Line Angle: {f'{abs(90 - line_angle):.2f}' if line_angle else 'N/A'}, "
            f"Final Angle: {f'{90 - final_angle:.2f}' if final_angle is not None else 'N/A'}"
        )
        


        angle_precision = max(0, int(angle_precision))

        cv2.putText(
            frame, 
            f"{f'{(90 - final_angle):+.{angle_precision}f}' if final_angle is not None else 'N/A'}",
            (50, 100),                
            cv2.FONT_HERSHEY_SIMPLEX, 
            2.0,                      
            (0, 255, 0),              
            4                         
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

    # --- CLEANUP VIDEO WINDOWS ---
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # <--- UI THREAD PATCH: Gives macOS a moment to fully close the OpenCV window loop
    cv2.waitKey(1)

    # --- GENERATE AND SAVE THE GRAPH AFTER VIDEO CLOSES ---
    clean_angle_log = [x for x in angle_history_log if not np.isnan(x)]

    if clean_angle_log:
        print("Rendering your Stroke Profile Graph to file...")
        
        plt.figure(figsize=(10, 5)) 
        
        # 1. Plot the data first
        plt.plot(clean_angle_log, color='blue', linewidth=2, label='Cue Angle')
        
        initial_aim_angle = clean_angle_log[0]

        # 2. Draw the baseline
        plt.axhline(y=initial_aim_angle, color='red', linestyle='--', alpha=0.6, label='Initial Line of Aim')
        
        # 3. FIX: Dynamically center the Y-axis limits FIRST using max() so the window is wide enough
        # We find the furthest absolute deviation from the initial aim angle
        deviations = [abs(x - initial_aim_angle) for x in clean_angle_log]
        max_deviation = max(max(deviations), 2.0) # Ensure a minimum window height of 2 degrees
        
        plt.ylim(initial_aim_angle - max_deviation - 0.5, initial_aim_angle + max_deviation + 0.5) 

        # 4. FIX: Let Matplotlib generate the final layout ticks *after* limits are set
        ax = plt.gca()
        # Trigger a draw behind the scenes so Matplotlib populates the real ticks for our new ylims
        plt.gcf().canvas.draw() 
        
        current_ticks = list(ax.get_yticks())

        # Only append if it's not already closely represented on the axis
        if not any(np.isclose(initial_aim_angle, tick, atol=0.2) for tick in current_ticks):
            current_ticks.append(initial_aim_angle)
            
        # Explicitly apply the combined list
        plt.yticks(current_ticks)

        # Add labels and styling
        plt.title("Snooker Stroke Profile Analysis", fontsize=14, fontweight='bold')
        plt.xlabel("Time (Frames)", fontsize=12)
        plt.ylabel("Deviation Angle (Degrees)\n← Bottom-Left  |  Bottom-Right →", fontsize=12)
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        
        # Save straight to your project folder
        output_filename = f"output/stroke_profile.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        
        print(f" Success! Your graph has been saved as '{output_filename}' in your project folder.")
    else:
        print("No valid tracking data to plot.")




if __name__ == "__main__":

    image_path = "Data/20260620 EB Birds Eye view first frame.png"

    frame = cv2.imread(image_path)
    # 2. Setup ROI
    roi = cv2.selectROI("Select ROI", frame, False)
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    cropped_first_frame = frame[y:y+h, x:x+w]

    engine = CueAngleEngine(buffer_size=5) 

    x_pos, y_pos = engine.get_position(cropped_first_frame)

    print(f"x of white: {x_pos}, y of white: {y_pos}")


    # video_path = "Data/20260620 EB Birds eye view 1.mp4"
    # pixel_threshold_range = list(range(100,256,1))
    # main(video_path=video_path,
    #      pixel_threshold_range=pixel_threshold_range,
    #      optimal_window_size=30,
    #      angle_precision=1,
    #      buffer_size=5)