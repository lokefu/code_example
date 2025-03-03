import os
import cv2
import shutil
import random
import re
from PIL import Image


def make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
    return folder

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
    return count #the number of frames saved

def generate_light_color(id):
    random.seed(id)
    # Generate random light color (higher brightness values)
    r = random.randint(0, 255)  # Red component
    g = random.randint(0, 255)  # Green component
    b = random.randint(0, 255)  # Blue component
    return (r, g, b)

def extract_frame_number(file_name):
    return int(re.search(r'\d+', file_name).group())

def create_gif_from_images(save_path : str, image_path : str, ext : str) -> None:
    ''' creates a GIF from a folder of images
        Inputs:
            save_path - path to save GIF
            image_path - path where images are located
            ext - extension of the images
        Outputs:
            None
    '''
    print('start to create GIF')
    # Sort the file names based on the extracted integer value
    image_paths = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')], key=extract_frame_number)
    pil_images = [Image.open(os.path.join(image_path, im_path)) for im_path in image_paths]
    #print(image_paths)

    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=50, loop=0)

def convert_images_to_video(input_folder, output_file, fps=30):
    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')], key=extract_frame_number)
    
    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer and close the video file
    video.release()
    cv2.destroyAllWindows()