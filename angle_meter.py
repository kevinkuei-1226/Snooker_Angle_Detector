import cv2
import numpy as np
import time

def draw_front_view_meter(angle, width=600, height=400):
    """
    Creates a dynamic angle meter for a FRONT VIEW camera layout.
    90 degrees is treated as the stable center line pointing STRAIGHT DOWN.
    """
    # Create a dark background canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8) + 20
    
    # NEW ANCHOR: Pivot point is now at the top-middle of the meter area
    center_x = width // 2
    center_y = int(height * 0.25)
    needle_length = int(height * 0.5)
    
    # Calculate deviation from the center line (90 degrees)
    deviation = angle - 90.0
    
    # Invert the direction: 
    # To point down, our base geometric angle is 270 degrees.
    # A positive deviation swings it left/right depending on preference.
    amplified_angle = 270.0 - (deviation * 3.0)  
    
    # Convert angle to radians
    # 270° is straight down. 
    # Subtracting the deviation makes a positive deviation swing right, negative swing left.
    rad = np.radians(270.0 - amplified_angle)
    
    # Calculate end coordinates pointing downwards from the top pivot
    end_x = int(center_x + needle_length * np.sin(rad))
    end_y = int(center_y + needle_length * np.cos(rad))
    
    # --- DYNAMIC GRADIENT COLORING ---
    max_alert_deviation = 15.0
    severity = min(abs(deviation) / max_alert_deviation, 1.0)
    
    if severity < 0.5:
        t = severity / 0.5
        b, g, r = 0, 255, int(255 * t)
    else:
        t = (severity - 0.5) / 0.5
        b, g, r = 0, int(255 * (1 - t)), 255
    
    color = (b, g, r)
    
    # --- DRAWING THE METER GRAPHICS ---
    # Draw reference arc background pointing downward
    cv2.ellipse(canvas, (center_x, center_y), (needle_length, needle_length), 0, 0, 180, (60, 60, 60), 2)
    
    # Draw center target line mark (Perfect alignment guide pointing straight down)
    cv2.line(canvas, (center_x, center_y + needle_length - 10), (center_x, center_y + needle_length + 10), (100, 100, 100), 2)
    
    # Draw the dynamic tracking needle
    cv2.line(canvas, (center_x, center_y), (end_x, end_y), color, 4, cv2.LINE_AA)
    cv2.circle(canvas, (center_x, center_y), 8, color, -1)
    cv2.circle(canvas, (end_x, end_y), 6, color, -1)
    
    # Text Overlays
    cv2.putText(canvas, f"Front View Angle: {angle:.1f} deg", (30, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    
    deviation_sign = "+" if deviation >= 0 else ""
    cv2.putText(canvas, f"Dev: {deviation_sign}{deviation:.1f}", (30, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    if severity >= 0.8:
        cv2.putText(canvas, "WARNING: CUE DRIFT", (width - 230, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
    return canvas

if __name__ == "__main__":
    # Simulate data swinging around 90 degrees
    simulated_angles = []
    for t in range(1000):
        if 120 <= t <= 140:
            simulated_angles.append(float('nan'))
        else:
            amplitude = 1.5 + (t * 0.04)
            mock_angle = 90.0 + amplitude * np.sin(t * 0.1)
            simulated_angles.append(mock_angle)

    print("Starting Front-View Angle Meter simulation. Press 'q' to exit.")
    
    last_known_good_angle = 90.0
    for angle in simulated_angles:
        if np.isnan(angle):
            meter_frame = draw_front_view_meter(last_known_good_angle)
            cv2.putText(meter_frame, "TRACKING LOST", (180, int(meter_frame.shape[0] * 0.6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            meter_frame = draw_front_view_meter(angle)
            last_known_good_angle = angle
            
        cv2.imshow("Front View Cue Balance", meter_frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()