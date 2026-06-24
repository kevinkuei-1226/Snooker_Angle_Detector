import cv2
import numpy as np

# Returns the contour from the image based on certain filtering methods
# roi: a frame object, representing a region of interest of an image
# method: a string, the method for filtering the image, currently supports "grayScale", "HSV"
# params: an array, relevant parameters needed for different methods, need not be the same length

# for "grayScale": params[0] = brightness threshold
#                  params[1] = 2-value-tuple of gaussian blur kernel size

# for "HSV":       params[0] = 3-value-tuple of HSV lower bound
#                  params[1] = 3-value-tuple of HSV upper bound
#                  params[2] = value for medianBlur

def get_Contours(roi,
                method,
                params):
    


    if method == "grayScale": # filters by brightness

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, params[1], 0) # smooth out wood grain
        _, thresh = cv2.threshold(blurred, params[0], 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    elif method == "HSV": # HSV filtering method

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_blurred = cv2.medianBlur(v, params[2])
        hsv_blurred = cv2.merge([h, s, v_blurred]) # merge blur and HSV separation

        # create mask using the blurred HSV data
        lower_color_bound = np.array(params[0]) 
        upper_color_bound = np.array(params[1])
        mask = cv2.inRange(hsv_blurred, lower_color_bound, upper_color_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    return contours


def get_frame_from_video(video_path,
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