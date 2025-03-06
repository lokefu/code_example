#pip install moviepy


#### cut out part of video
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
def cut_video(input_video, output_video, start_time, end_time):
    clip = (
        VideoFileClip(input_video)
        .subclipped(start_time, end_time) #between in seconds
        .with_volume_scaled(0)
    )

    final_video = CompositeVideoClip([clip])
    final_video.write_videofile(output_video)


#### cut video into a series of segments
from moviepy import VideoFileClip, CompositeVideoClip
import os
import shutil

def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
    return folder

def seg_video(video, output_folder, segment_duration=10):
    # Define the input video file path
    input_video_path = video #"data/1.mp4"

    # Define the output folder where the segmented videos will be saved
    #output_folder = "segmented_videos/"
    make_folder(output_folder)
    
    # Time duration for each video segment (in seconds)
    #segment_duration = 1

    # Open the input video file
    original_clip = VideoFileClip(input_video_path)

    # Calculate the total duration of the video
    total_duration = original_clip.duration

    # Iterate over the video and create segments
    start_time = 0
    segment_number = 1

    while start_time < total_duration:
        end_time = min(start_time + segment_duration, total_duration)
        
        # Subclip the video based on the start and end times
        clip = original_clip.subclipped(start_time, end_time).with_volume_scaled(0.8)
        
        # Create a CompositeVideoClip with the subclip
        final_video = CompositeVideoClip([clip])
        
        # Define the output file path for the segmented video
        output_file_path = f"{output_folder}{segment_number}.mp4"
        
        # Write the segmented video to the output file
        final_video.write_videofile(output_file_path)
        
        # Update the start time and segment number for the next iteration
        start_time = end_time
        segment_number += 1


#### lower video fps
import random
import cv2
import os
import shutil

def generate_random_with_probability(true_probability):
    if random.random() < true_probability:
        return True
    else:
        return False

def convert_video_to_images_random_cut(input_video, output_folder, prob=0.5):
    # Create output folder if it doesn't exist
    
    from .utils import make_folder
    output_folder = make_folder(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(input_video)
    success, frame = video_capture.read()
    count = 0

    # Read each frame and save it as an image
    while success: # and count < 1000:
        image_path = os.path.join(output_folder, f"frame_{count:d}.jpg")  # Adjust the format as per your requirement
        if generate_random_with_probability(prob):
            cv2.imwrite(image_path, frame)  # Save the frame as an image
            count += 1
        success, frame = video_capture.read()  # Read next frame

    # Release the video capture object
    video_capture.release()
    return count #the number of frames saved (good/successful frames, not the whole frame count)

def lower_fps(input_video, img_folder, output_video, new_fps):
    print("Original FPS:")
    
    from .video_info import get_fps
    old_fps = get_fps(input_video)
    prob = new_fps / old_fps
    convert_video_to_images_random_cut(input_video, img_folder, prob)
    
    from .img_vid import convert_images_to_video
    convert_images_to_video(img_folder, output_video, new_fps)


#### merge videos
from moviepy import VideoFileClip, concatenate_videoclips

def merge_video(input_video_list, output_path):
    # Load all video clips
    clips = [VideoFileClip(path) for path in input_video_list]

    # Concatenate all video clips
    final_clip = concatenate_videoclips(clips)

    # Write the merged video to a file
    final_clip.write_videofile(output_path)