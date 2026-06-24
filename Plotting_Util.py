import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_results(CAS,
                 angle_log):
    if angle_log:
        print("Rendering your Stroke Profile Graph to file...")
        
        # 1. Initialize the main figure and primary axis (ax1)
        fig, ax1 = plt.subplots(figsize=(10, 5)) 
        
        # --- PRIMARY AXIS (Y1): CUE ANGLE ---
        # Plot the primary data line
        line1 = ax1.plot(angle_log, color='blue', linewidth=2, label='Cue Angle')
        
        initial_aim_angle = angle_log[0]
        # Draw the baseline
        ax1.axhline(y=initial_aim_angle, color='blue', linestyle='--', alpha=0.4)
        
        # Centering the primary Y-axis limits
        deviations = [abs(x - initial_aim_angle) for x in angle_log]
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