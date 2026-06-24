import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Cue_Angle_Session import Cue_Angle_Session # <--- This is the link!


def main(video_path, pixel_threshold_range, optimal_window_size, buffer_size=1, angle_precision=0):
    # 1. Initial Setup
    CAS = Cue_Angle_Session(method="grayScale",
                            buffer_size=buffer_size
                            ) 
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    roi = cv2.selectROI("Select ROI", frame, False)
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    cropped_first_frame = frame[y:y+h, x:x+w]

    # 2. Calibrate Threshold DOBLE CHECK THIS SYNTAXX
    opt_threshold_stats = CAS.find_optimal_threshold(selected_roi=cropped_first_frame,
                                                        threshold_range=pixel_threshold_range,
                                                        window_size=optimal_window_size)

    opt_threshold = opt_threshold_stats['start'] + int(optimal_window_size)

    frame_num = 1

    # 3. Read frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 3.1 tracking physics of white paper
        
        x_pos, y_pos = CAS.get_position(image=frame[y:y+h, x:x+w])
        box_angle, line_angle = CAS.get_angle(image=frame[y:y+h, x:x+w])
        

        # 3.2. Smooth them using the engine's internal memory
        box_angle_smoothed, line_angle_smoothed, position_smoothed = CAS.get_smoothed_values()
        

        # 3.3 Fusion Filtering
        # checks consistency between box and line and determine whether the output is valid

        if not box_angle_smoothed or not line_angle_smoothed:
            final_angle = None
        elif box_angle_smoothed == 0.00 or box_angle_smoothed == 90.00:
            # Case A: The box has snapped to the grid lines (blind spot). Rely entirely on the line.
            final_angle = 90 - line_angle_smoothed
        elif abs(box_angle_smoothed - line_angle_smoothed) > 10.0:
            # Case B: They wildly disagree. When moving near vertical, the box's structural axis 
            # anchor is typically more stable than fitLine's pixel slope regression.
            final_angle = 90 - box_angle_smoothed
        else:
            # Case C: They are clean and agree. Average them together to eliminate individual noise!
            final_angle = 90 - (box_angle_smoothed + line_angle_smoothed) / 2.0


        # 3.4 update session history
        CAS.update_history(new_box_angle=box_angle,
                           new_line_angle=line_angle,
                           new_position=(x_pos, y_pos),
                           new_final_angle=final_angle)

        # print(
        #     f"frame number: {frame_num}"
        #     f"Unsmoothed Box Angle: {f'{90 - box_angle:.2f}' if box_angle else 'N/A'}, "
        #     f"Unsmoothed Line Angle: {f'{abs(90 - line_angle):.2f}' if line_angle else 'N/A'}, "
        #     f"Final Angle: {f'{90 - final_angle:.2f}' if final_angle is not None else 'N/A'}"
        # )

        frame_num += 1
        
        

        angle_precision = max(0, int(angle_precision))

        cv2.putText(
            frame, 
            f"{f'{(final_angle):+.{angle_precision}f}' if final_angle is not None else 'N/A'}",
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

    print(f"line_history: {CAS.line_history}")

    # --- CLEANUP VIDEO WINDOWS ---
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # <--- UI THREAD PATCH: Gives macOS a moment to fully close the OpenCV window loop
    cv2.waitKey(1)

    # --- GENERATE AND SAVE THE GRAPH AFTER VIDEO CLOSES ---
    clean_angle_log = [x for x in CAS.final_output_history if not np.isnan(x)]

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
        raw_y_positions = [pos[1] for idx, pos in enumerate(CAS.position_history) if not np.isnan(CAS.final_output_history[idx])]
        
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

    video_path = "Data/20260620 EB Birds eye view 1.mp4"
    pixel_threshold_range = list(range(100,256,1))
    main(video_path=video_path,
         pixel_threshold_range=pixel_threshold_range,
         optimal_window_size=30,
         angle_precision=1,
         buffer_size=5)
    
