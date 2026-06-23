import cv2


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