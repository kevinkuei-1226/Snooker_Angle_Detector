import cv2
from cue_engine import CueAngleEngine # <--- This is the link!

def main(video_path):
    # 1. Initialize the engine
    engine = CueAngleEngine() 
    
    print("created engine")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    print("read video")
    # 2. Setup ROI
    roi = cv2.selectROI("Select ROI", frame, False)
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 3. Call the engine
        box_angle, line_angle = engine.get_angle(frame[y:y+h, x:x+w])
        
        if angle is not None:
            print(f"Angle: {angle:.2f}")
            cv2.putText(frame, f"{(90 - angle):.2f}", (50,50), 1, 2, (0,255,0))
            
        cv2.imshow("Main View", frame)
        if cv2.waitKey(1) == ord('q'): break

if __name__ == "__main__":
    video_path = "Data/Cue with paper between head and hand 1.mp4"
    main(video_path=video_path)