#### make folder
import os
import shutil

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
    return folder

#### load from json
import json
def load(path):
    # Load the JSON file
    with open(path, 'r') as json_file:
        return json.load(json_file)

#### save into json
import json
def save(path, data):    
    # Save the dictionary to a JSON file
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

#### get all folders with prefix
import os

def get_folders_with_prefix(directory_path, prefix):
    """
    Get all folder names inside a directory that start with a specific prefix.

    Args:
        directory_path (str): Path to the directory to search in.
        prefix (str): Prefix that the folder names should start with.

    Returns:
        list: List of folder names that start with the specified prefix.
    """
    folder_names = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder)) and folder.startswith(prefix)]
    return folder_names

#### copy file
import shutil

def copy_file(source_path, destination_path):
    """
    Copy a file from the source path to the destination path.

    Args:
        source_path (str): Path to the source file.
        destination_path (str): Path to the destination directory.

    Returns:
        bool: True if the file is successfully copied, False otherwise.
    """
    try:
        shutil.copy(source_path, destination_path)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

#### cut out part of video
#pip install moviepy
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
def cut_video(input_video, output_video, start_time, end_time):
    clip = (
        VideoFileClip(input_video)
        .subclipped(start_time, end_time) #between in seconds
        .with_volume_scaled(0)
    )

    final_video = CompositeVideoClip([clip])
    final_video.write_videofile(output_video)
#cut_video("data/1.mp4", "data/1_cut.mp4", 0, 10)

#### convert video to images
import cv2
def convert_video_to_images(input_video, output_folder):
    # Create output folder if it doesn't exist
    output_folder = make_folder(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(input_video)
    success, frame = video_capture.read()
    count = 0

    # Read each frame and save it as an image
    while success: # and count < 1000:
        image_path = os.path.join(output_folder, f"frame_{count:d}.jpg")  # Adjust the format as per your requirement
        cv2.imwrite(image_path, frame)  # Save the frame as an image
        success, frame = video_capture.read()  # Read next frame
        count += 1

    # Release the video capture object
    video_capture.release()
    return count #the number of frames saved (good/successful frames, not the whole frame count)