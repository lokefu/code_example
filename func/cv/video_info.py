#pip install opencv-python


#### video information
import cv2

def get_image_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = frame_count / fps
    
    cap.release()
    return frame_count, fps, video_length


#### count frames
import cv2

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print("Number of frames in the video:", frame_count)

    cap.release()
    return frame_count

# Specify the path to your video file
#video_path = "yes/1_yes_1.mp4"

#count_frames(video_path)


#### fps: frame_per_second
import cv2

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    #print(f"Frames per second (fps): {fps}")
    
    cap.release()
    return fps

# Example usage
#video_path = "yes/1_yes_1.mp4"
#get_fps(video_path)


#### video length
import cv2

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate video length in seconds
    video_length = frame_count / fps
    #print(f"Video length: {video_length} seconds")
    
    cap.release()
    
    return video_length