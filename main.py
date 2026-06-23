import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cue_engine import CueAngleEngine # <--- This is the link!

def get_frame(video_path,
              target_frame
              ):

    cap = cv2.VideoCapture(video_path)

    # 1. Jump to the targeted frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # 2. Read the frame
    ret, frame = cap.read()

    cap.release()

    if ret:
        # 3. Define your output path and save it as a PNG
        output_filename = f"output/frame_{target_frame}.png"
        
        # cv2.imwrite saves the array data straight to a file
        success = cv2.imwrite(output_filename, frame)
        
        if success:
            print(f"Success! Saved frame {target_frame} to '{output_filename}'")
        else:
            print("Error: The frame was read, but could not be saved to disk. Check your directory permissions.")
    else:
        print(f"Error: Could not read frame {target_frame}.")


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
    opt_threshold = opt_threshold_stats['start'] + int(optimal_window_size)

    print(f"optimal threshold: {opt_threshold}")

    angle_history_log = []
    Position_log = []
    frame_num = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # tracking position of the white paper
        
        x_pos, y_pos = engine.get_position(image=frame[y:y+h, x:x+w])
        Position_log.append([x_pos, y_pos])

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
            f"frame number: {frame_num}"
            f"Unsmoothed Box Angle: {f'{90 - box_angle:.2f}' if box_angle else 'N/A'}, "
            f"Unsmoothed Line Angle: {f'{abs(90 - line_angle):.2f}' if line_angle else 'N/A'}, "
            f"Final Angle: {f'{90 - final_angle:.2f}' if final_angle is not None else 'N/A'}"
        )

        frame_num += 1
        


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
        
        # 1. Initialize the main figure and primary axis (ax1)
        fig, ax1 = plt.subplots(figsize=(10, 5)) 
        
        # --- PRIMARY AXIS (Y1): CUE ANGLE ---
        # Plot the primary data line
        line1 = ax1.plot(clean_angle_log, color='blue', linewidth=2, label='Cue Angle')
        
        initial_aim_angle = clean_angle_log[0]
        # Draw the baseline
        ax1.axhline(y=initial_aim_angle, color='blue', linestyle='--', alpha=0.4)
        
        # Centering the primary Y-axis limits
        deviations = [abs(x - initial_aim_angle) for x in clean_angle_log]
        max_deviation = max(max(deviations), 2.0) 
        ax1.ylim_val = (initial_aim_angle - max_deviation - 0.5, initial_aim_angle + max_deviation + 0.5)
        ax1.set_ylim(ax1.ylim_val)

        # Add labels for primary axis
        ax1.set_title("Snooker Stroke Profile & Position Analysis", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Time (Frames)", fontsize=12)
        ax1.set_ylabel("Deviation Angle (Degrees)\n← Bottom-Left  |  Bottom-Right →", color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle=':', alpha=0.6)

        # Let Matplotlib generate layout ticks, then append initial aim
        plt.gcf().canvas.draw() 
        current_ticks = list(ax1.get_yticks())
        if not any(np.isclose(initial_aim_angle, tick, atol=0.2) for tick in current_ticks):
            current_ticks.append(initial_aim_angle)
        ax1.set_yticks(current_ticks)

        # --- SECONDARY AXIS (Y2): STANDARDIZED POSITION ---
        # Filter Position_log to match frame-for-frame with non-nan angles if necessary, 
        # but extracting Y-positions here assuming consecutive frame tracking:
        raw_y_positions = [pos[1] for idx, pos in enumerate(Position_log) if not np.isnan(angle_history_log[idx])]
        
        if raw_y_positions:
            # Standardize: Set initial position as 0, map relative drift max step to 1.0/-1.0 range
            initial_y = raw_y_positions[0]
            y_offsets = [y - initial_y for y in raw_y_positions]
            max_offset = max(max([abs(o) for o in y_offsets]), 1.0) # avoid division by zero
            
            # Scale to [-1.0, 1.0]
            standardized_y = [o / max_offset for o in y_offsets]
            
            # Create the twin y-axis sharing the same x-axis
            ax2 = ax1.twinx()
            line2 = ax2.plot(standardized_y, color='purple', linewidth=1.5, linestyle='-.', label='Sleeve Y-Pos (Normalized)')
            
            ax2.set_ylabel("Standardized Sleeve Position\n← Closer to Bridge  |  Closer to Grip →", color='purple', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='purple')
            ax2.set_ylim(-1.2, 1.2) # Set clean structural boundary bounds for [-1, 1] data
            
            # Combine legends from both axes seamlessly
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
        else:
            ax1.legend(loc='upper right')
        
        # Save straight to your project folder
        output_filename = f"output/stroke_profile.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        
        print(f" Success! Your graph has been saved as '{output_filename}' in your project folder.")
    else:
        print("No valid tracking data to plot.")




if __name__ == "__main__":

    # video_path = "Data/20260620 EB Birds eye view 1.mp4"
    # pixel_threshold_range = list(range(100,256,1))
    # main(video_path=video_path,
    #      pixel_threshold_range=pixel_threshold_range,
    #      optimal_window_size=30,
    #      angle_precision=1,
    #      buffer_size=5)
    
    get_frame(video_path="Data/20260620 EB Birds eye view 1.mp4",
              target_frame=11
    )
