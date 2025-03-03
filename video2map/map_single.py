import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import json
import glob
from PIL import Image
import re
import random
from util import *


#### mapping, homography

def map(coord_src, coord_map):
    # Load images
    #coord: four sets of coordinates in the source image and the map
    #coord_src = [[450, 580], [930, 815], [890, 400], [1340, 520]]
    #coord_map = [[290, 270], [290, 500], [590, 270], [590, 500]]

    # Find the homography matrix using RANSAC
    H, _ = cv2.findHomography(np.array(coord_src), np.array(coord_map), cv2.RANSAC, 5.0)

    return H


#### read json

# Function to read data from a JSON file, convert string values to integers, and sort based on the first two dimensions
def read_and_sort_json_file(file_path, H):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert string values to integers and sort based on the first two dimensions
    tmp = {}
    for key, value in data.items():
        converted_value = {int(k): [int(item) for item in v] for k, v in value.items()}
        sorted_value = dict(sorted(converted_value.items(), key=lambda x: x[0]))
        tmp[int(key)] = sorted_value

    sorted_data = dict(sorted(tmp.items(), key=lambda x: x[0]))
    first_frame_num = next(iter(sorted_data.keys()))
    
    data = {}
    # Calculate the average of specific values in the list and convert them to integers
    for key, value in sorted_data.items():
        tmp = {}
        for inner_key, inner_value in value.items():
            #center
            #pts_src = [int(inner_value[0] + inner_value[2]/2), int(inner_value[1] + inner_value[3]/2)]
            #bottom_mid, xy at the down-left
            pts_src = [int(inner_value[0] + inner_value[2]/2), int(inner_value[1] + inner_value[3])]
            #x,y at the top-left
            #pts_src = [int(inner_value[0] + inner_value[2]/2), int(inner_value[1])]
            pts_map = trans(pts_src, H)
            tmp[inner_key] = pts_map
        data[key] = tmp

    box = {}
    # Calculate the average of specific values in the list and convert them to integers
    for key, value in sorted_data.items():
        tmp = {}
        for inner_key, inner_value in value.items():
            #center
            #pts_src = [int(inner_value[0] + inner_value[2]/2), int(inner_value[1] + inner_value[3]/2)]
            #bottom_mid
            boxx = [int(inner_value[0]), int(inner_value[1]), int(inner_value[0] + inner_value[2]), int(inner_value[1] + inner_value[3])]
            tmp[inner_key] = boxx
        box[key] = tmp
    return data, first_frame_num, box

def trans(src_pts, H):
    #src_pts = [1350, 600]
    arr = np.array(src_pts, dtype=np.float32)
    arr_tmp = arr.reshape(-1, 1, 2)
    dst_pts = cv2.perspectiveTransform(arr_tmp, H)
    for point in dst_pts:
        x, y = point[0]
        return int(x), int(y)


#### drawing

def color(data):
    unique_ids = set()  # Set to store unique IDs
    id_colors = {}  # Dictionary to store colors for each ID

    # Loop through the data structure to count unique IDs and assign random colors
    for outer_key in data:
        for inner_key in data[outer_key]:
            unique_ids.add(inner_key)

    # Assign random colors to each unique ID
    for id_num in unique_ids:
        id_colors[id_num] = generate_light_color(id_num)
    
    return id_colors #dictionary

def draw_bboxes(frame, det, color_dict):
    for obj_id, det in det.items():
        color = color_dict[obj_id]
        x1,y1,x2,y2 = det[0], det[1], det[2], det[3]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3) #(0,255,0), 3) green
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)  # Add ID label above the rectangle

# Function to draw line tracking for each object over 2 frames with different line styles
def draw_tracking(frame, current_object_data, prev_object_data, prev_prev_object_data, p_triple, color_dict):
    for obj_id, (x, y) in current_object_data.items():
        color = color_dict[obj_id]
        if obj_id in prev_object_data:
            prev_x, prev_y = prev_object_data[obj_id]
            cv2.line(frame, (prev_x, prev_y), (x, y), color, 3)  # Draw a line between previous and current position with regular line thickness
            if  obj_id in prev_prev_object_data:
                prev_prev_x, prev_prev_y = prev_prev_object_data[obj_id]
                cv2.line(frame, (prev_prev_x, prev_prev_y), (prev_x, prev_y), color, 2)  # Draw a line between 2nd previous and previous position with thinner line
                if obj_id in p_triple:
                    prev_prev_prev_x, prev_prev_prev_y = p_triple[obj_id]
                    cv2.line(frame, (prev_prev_prev_x, prev_prev_prev_y), (prev_prev_x, prev_prev_y), color, 1)
        cv2.circle(frame, (x, y), 5, color, -1)  # Draw a circle at the current position
        cv2.putText(frame, f'ID: {obj_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame

def draw(data, box, idx, first_frame_num, map_path, color_dict, map_folder, box_folder):
    '''
    #map_path = 'map_indoor.png'
    frame = cv2.imread(map_path)
    fig = plt.figure(figsize=(15, 7))
    plt.imshow(frame)
    plt.axis('off')
    fig.savefig('in_back')
    plt.close()
    '''

    # Initialize previous object data
    prev_object_data = {}
    prev_prev_object_data = {}
    p_triple = {}

    folder = map_folder #'temp_in'
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    print('map')
    for frame_number, object_data in data.items():
        if int(frame_number) > idx:
            break
        #print(f"Frame {frame_number}:")
        num = int(frame_number)
        #frame = cv2.imread(f'src_out/frame_{num}.jpg')  # Load the frame for src_img
        frame = cv2.imread(map_path)
        if num == first_frame_num:
            p_triple = object_data
            prev_prev_object_data = object_data
            prev_object_data = object_data
            current_object_data = object_data
        else:
            p_triple = prev_prev_object_data
            prev_prev_object_data = prev_object_data
            prev_object_data = current_object_data
            current_object_data = object_data

        # Draw line tracking for each object
        frame_with_tracking = draw_tracking(frame, current_object_data, prev_object_data, prev_prev_object_data, p_triple, color_dict)

        # save image for GIF
        fig = plt.figure(figsize=(15, 7))
        plt.imshow(frame_with_tracking)
        plt.axis('off')
        fig.savefig(f"{folder}/frame_{frame_number}.jpg")
        plt.close()
    
    #boxxes
    print('box')
    folder = box_folder #'temp_in_box'
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

    for frame_number, object_data in box.items():
        if int(frame_number) > idx:
            break
        #print(f"Frame {frame_number}:")
        num = int(frame_number)
        frame = cv2.imread(f'src_in/frame_{num}.jpg')  # Load the frame for src_img
        
        draw_bboxes(frame, object_data, color_dict)
        # save image for GIF
        fig = plt.figure(figsize=(15, 7))
        plt.imshow(frame)
        plt.axis('off')
        fig.savefig(f"{folder}/frame_{frame_number}.jpg")
        plt.close()
    
    '''
    # Source folder to be copied
    source_folder = map_folder

    # Destination folder for the copy
    destination_folder = f'map_folder{2}'
    if not os.path.exists(destination_folder):
        pass
    else:
        shutil.rmtree(destination_folder)

    # Copy the folder and its contents
    shutil.copytree(source_folder, destination_folder)

    print(f'Folder copied from {source_folder} to {destination_folder}')

    # Path to the sample image
    map = 'in_back.png'

    # Generate filenames and save images
    for i in range(idx+1):  # Range from 0 to 426 inclusive
        new_filename = f'frame_{i}.jpg'
        new_image_path = os.path.join(destination_folder, new_filename)
        
        if os.path.exists(new_image_path):
            continue
        else:
            shutil.copy2(map, new_image_path)
    '''


#### create gif

def merge(box_folder, map_folder, idx, dpi):
    folder = 'merge'
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
    
    # Load the two images
    for i in range(idx):
        image1 = cv2.imread(f'{box_folder}/frame_{i}.jpg')
        image2 = cv2.imread(f'{map_folder}/frame_{i}.jpg')

        # Create an image with the dimensions to accommodate both images
        max_width = max(image1.shape[1], image2.shape[1])
        total_height = image1.shape[0] + image2.shape[0]
        image_combined = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        # Place the first image at the top of the combined image
        image_combined[:image1.shape[0], :image1.shape[1]] = image1

        # Place the second image below the first image
        image_combined[image1.shape[0]:, :image2.shape[1]] = image2
        
        fig = plt.figure(figsize=(20, 14))
        plt.imshow(cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        fig.savefig(f"{folder}/frame_{i}.jpg", dpi=dpi)
        plt.close()
    
    return folder


#### main

def main(coord_src, coord_map, map_path, video, json_file_path, output_format, output_filename, fps, map_folder, box_folder, dpi):
    print('compute homography matrix')
    H = map(coord_src, coord_map)
    print('convert video to images')
    count = convert_video_to_images(video, 'src_in')
    idx = count - 1
    print('read and sort json file')
    data, first_frame_num, box = read_and_sort_json_file(json_file_path, H)
    print('draw')
    color_dict = color(data)
    draw(data, box, idx, first_frame_num, map_path, color_dict, map_folder, box_folder)
    print('merge')
    folder = merge(box_folder, map_folder, idx+1, dpi)
    if output_format == 'gif':
        print('create gif')
        output_path = output_filename + '.gif'
        create_gif_from_images(output_path, folder, '.png')
    else:
        print('convert images to video')
        output_path = output_filename + '.mp4'
        convert_images_to_video(folder, output_path, fps)
