#python 3.9.2
#use conda virtual environment, no specific requirement for libraries' versions

import os
from glob import glob
import re
import numpy as np
import cv2
from PIL import Image


import argparse
import shutil

def convert_video_to_images(input_video, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

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

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='inputs')

# Add arguments
parser.add_argument('--thresh', type=int, default=1000, required=False, help='box size lower limit')
parser.add_argument('--video_path', type=str, default='highway_cut.mp4', required=False, help='video_path end with .mp4')
parser.add_argument('--output_file', type=str, default='data.csv', required=False, help='box output file ending with csv')

# Parse the arguments
args = parser.parse_args()
thresh = args.thresh
video_path = args.video_path
# Call the function to convert the video to images
convert_video_to_images(video_path, 'Frame')


# data_path is the folder name without / at the end
# assume the folder contains only a numerical-ordered number of images ended with jpg
# e.g., ['10s_Frame/frame_0.jpg', '10s_Frame/frame_1.jpg', '10s_Frame/frame_2.jpg']

data_path = 'Frame' 
image_paths = sorted(glob(f"{data_path}/*.jpg"), key=lambda x: int(re.search(r'(\d+).jpg', os.path.basename(x)).group(1)))


# =============================================================================
# get bounding box detections from blobs/contours

def get_mask(frame1, frame2, kernel=np.array((9,9), dtype=np.uint8)):
    """ Obtains image mask
        Inputs: 
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            kernel - (NxN) array for Morphological Operations
        Outputs: 
            mask - Thresholded mask for moving pixels
        """
    frame_diff = cv2.subtract(frame2, frame1)

    # blur the frame difference
    frame_diff = cv2.medianBlur(frame_diff, 3)
    
    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV, 11, 3)

    mask = cv2.medianBlur(mask, 3)

    # morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

def get_contour_detections(mask, thresh=400):
    """ Obtains initial proposed detections from contours discoverd on the mask. 
        Scores are taken as the bbox area, larger is higher.
        Inputs:
            mask - thresholded image mask
            thresh - threshold for contour size
        Outputs:
            detectons - array of proposed detection bounding boxes and scores [[x1,y1,x2,y2,s]]
        """
    # get mask contours
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_TC89_L1)
    detections = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area > thresh: # hyperparameter
            detections.append([x,y,x+w,y+h, area])
    
    if detections:  # Check if detections list is not empty before conversion
        return np.array(detections)
    else:
        return np.zeros((0, 5)) #np.empty((0, 5))  # Return an empty array if no detections meet the criteria
    #return np.array(detections)


# =============================================================================
# Non-Max Supression for detected bounding boxes on blobs

def remove_contained_bboxes(boxes):
    """ Removes all smaller boxes that are contained within larger boxes.
        Requires bboxes to be soirted by area (score)
        Inputs:
            boxes - array bounding boxes sorted (descending) by area 
                    [[x1,y1,x2,y2]]
        Outputs:
            keep - indexes of bounding boxes that are not entirely contained 
                   in another box
        """
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep: # range(0, len(bboxes)):
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep


def non_max_suppression(boxes, scores, threshold=1e-1):
    """
    Perform non-max suppression on a set of bounding boxes 
    and corresponding scores.
    Inputs:
        boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
        scores: a list of corresponding scores 
        threshold: the IoU (intersection-over-union) threshold for merging bboxes
    Outputs:
        boxes - non-max suppressed boxes
    """
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]

    # remove all contained bounding boxes and get ordered index
    order = remove_contained_bboxes(boxes)

    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)
                
    return boxes[keep]


def get_detections(frame1, frame2, bbox_thresh=400, nms_thresh=1e-3, mask_kernel=np.array((9,9), dtype=np.uint8)):
    """ Main function to get detections via Frame Differencing
        Inputs:
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            bbox_thresh - Minimum threshold area for declaring a bounding box 
            nms_thresh - IOU threshold for computing Non-Maximal Supression
            mask_kernel - kernel for morphological operations on motion mask
        Outputs:
            detections - list with bounding box locations of all detections
                bounding boxes are in the form of: (xmin, ymin, xmax, ymax)
        """
    # get image mask for moving pixels
    mask = get_mask(frame1, frame2, mask_kernel)

    # get initially proposed detections from contours
    detections = get_contour_detections(mask, bbox_thresh)

    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]
    
    if bboxes.size > 0:
        return bboxes
    else:
        # perform Non-Maximal Supression on initial detections
        return non_max_suppression(bboxes, scores, nms_thresh)


  
kernel=np.array((9,9), dtype=np.uint8)

box_result = []
for idx in range(1, len(image_paths)):
    # read frames
    frame1_bgr = cv2.imread(image_paths[idx - 1])
    frame2_bgr = cv2.imread(image_paths[idx])

    # get detections
    detections = get_detections(cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY), 
                                cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY), 
                                bbox_thresh=thresh,
                                nms_thresh=1e-4)
    box_result.append(detections)
print('the box thresh: ', thresh)  
'''
# example box output: list of array: an array is for a image, an list in the array is for a box; 
# box format: [xmin, ymin, xmax, ymax]
[array([[ 767,  596,  926,  750],
        [ 130,  372,  248,  449],
        [1142,  195, 1223,  267],
        [ 701,  127,  779,  180],
        [ 845,   34,  911,   95],
        [ 995,    0, 1053,   41]]),
 array([[ 766,  599,  926,  755],
        [ 135,  370,  255,  447],
        [1142,  196, 1222,  269],
        [ 846,   29,  913,   93],
        [ 705,  125,  781,  178],
        [ 997,    0, 1055,   39]])]
'''

# Save the list of arrays to a CSV file


file_path =  args.output_file # "data.csv"

# Check if the file exists before attempting to delete it
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been deleted.")
else:
    print(f"{file_path} does not exist. Create a new one")


with open(file_path, 'w') as file:
    for idx, arr in enumerate(box_result):
        np.savetxt(file, arr, delimiter=',', fmt='%d')
        if idx < len(box_result) - 1:
            file.write(';\n')  # Add a ; between arrays, except for the last one
print('the output file: ', file_path)        
'''
# how to import the csv
import numpy as np

# Initialize an empty list to store arrays
box_result = []

# Read the CSV file
with open('data.csv', 'r') as file:
    data = file.read().split(';\n')  # Split the file content by the word 'WORD'

    # Process each part of the split content
    for part in data:
        if part.strip():  # Check if the part is not empty
            arr = np.genfromtxt(part.splitlines(), delimiter=',', dtype=int)
            #box_result.append(arr)
        else:
            arr = np.zeros((0, 4))#'No Diff'
        box_result.append(arr)
# Print the list of arrays
for arr in box_result:
    print(arr)

'''


############################################


#visualize
from PIL import Image
import matplotlib.pyplot as plt


def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)


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
    ext = ext.replace('.', '')
    image_paths = sorted(glob(os.path.join(image_path, f'*.{ext}')))
    image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    pil_images = [Image.open(im_path) for im_path in image_paths]

    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=50, loop=0)

#'''
# Create output folder if it doesn't exist
import shutil
if not os.path.exists('temp'):
    os.makedirs('temp')
else:
    shutil.rmtree('temp')
    os.makedirs('temp')
    
print('start to visualize the box on the images')

for idx in range(1, len(image_paths)):
    # read frames
    frame_bgr = cv2.imread(image_paths[idx])
    detections = box_result[idx-1]                           
    # draw bounding boxes on frame
    draw_bboxes(frame_bgr, detections)

    # save image for GIF
    fig = plt.figure(figsize=(15, 7))
    plt.imshow(frame_bgr)
    plt.axis('off')
    fig.savefig(f"temp/frame_{idx}.png")
    plt.close()

file_path = "vis.GIF"

# Check if the file exists before attempting to delete it
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been deleted.")
else:
    print(f"{file_path} does not exist.")

create_gif_from_images(f"vis.GIF", 'temp', '.png')

#'''