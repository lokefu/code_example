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
from collections import defaultdict

from util import *


#### mapping, homography

def map(p_m_list):
    
    #p1 = [[750, 750], [870, 780], [1435, 375], [1550, 400]]
    #m1 = [[280, 600], [280, 710], [1520, 600], [1520, 710]]
    #p2, m2, p3, m3
    #p_m_list = [[p1, m1], [p1, m1], [p1, m1]]
    
    # Find the homography matrix using RANSAC
    H_list = []
    for i in p_m_list:
        H, _ = cv2.findHomography(np.array(i[0]), np.array(i[1]), cv2.RANSAC, 5.0)
        H_list.append(H)
    
    return H_list


#### functions

def get_frames(video_list):
    count_list = []
    frame_folder_list = []
    for idx, video in enumerate(video_list):
        frame_folder = f"video_{idx}"
        frame_folder_list.append(frame_folder)
        count = convert_video_to_images(video, frame_folder)
        count_list.append(count)
    count = min(count_list)
    return count, frame_folder_list


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

def get_data(box_list, H_list):
    data_list = []
    first_frame_num_list = []
    box_data_list = []
    for i in range(len(box_list)):
        data, first_frame_num, box = read_and_sort_json_file(box_list[i], H_list[i])
        data_list.append(data)
        first_frame_num_list.append(first_frame_num)
        box_data_list.append(box)
    #data_list: [{}, {}, {}]
    #first_frame_num_list: [0, 0, 0]
    #box_data_list: [{}, {}, {}]
    
    return data_list, first_frame_num_list, box_data_list

def merge_map_data(data_list):
    # Merged dictionary to store averaged coordinates
    merged_dict = defaultdict(lambda: defaultdict(list))

    # Iterate through each dictionary in the list
    for dictionary in data_list:
        for frame_no, objects in dictionary.items():
            for object_id, coordinates in objects.items():
                merged_dict[frame_no][object_id].append(coordinates)

    # Calculate the average coordinates for each object ID
    averaged_dict = {}
    for frame_no, objects in merged_dict.items():
        for object_id, coordinates_list in objects.items():
            if len(coordinates_list) == 1:
                averaged_dict.setdefault(frame_no, {})[object_id] = coordinates_list[0]
            else:
                avg_x = sum(x for x, _ in coordinates_list) / len(coordinates_list)
                avg_y = sum(y for _, y in coordinates_list) / len(coordinates_list)
                averaged_dict.setdefault(frame_no, {})[object_id] = (int(avg_x), int(avg_y))
    
    return averaged_dict

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

def draw_map(data, idx, first_frame_num, map_path, color_dict, map_folder):
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

    folder = make_folder(map_folder)

    #print('map')
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
        cv2.imwrite(f"{folder}/frame_{frame_number}.jpg", frame_with_tracking)
        # save image for GIF
        #fig = plt.figure(figsize=(15, 7))
        #plt.imshow(frame_with_tracking)
        #plt.axis('off')
        #fig.savefig(f"{folder}/frame_{frame_number}.jpg")
        #plt.close()
    
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

def draw_box(box, idx, color_dict, frame_folder, output_folder):
    #boxxes
    #print('box')
    folder = make_folder(output_folder) #'temp_in_box'

    for frame_number, object_data in box.items():
        if int(frame_number) > idx:
            break
        #print(f"Frame {frame_number}:")
        num = int(frame_number)
        frame = cv2.imread(f'{frame_folder}/frame_{num}.jpg')  # Load the frame for src_img
        
        draw_bboxes(frame, object_data, color_dict)
        cv2.imwrite(f"{folder}/frame_{frame_number}.jpg", frame)
        # save image for GIF
        #fig = plt.figure(figsize=(15, 7))
        #plt.imshow(frame)
        #plt.axis('off')
        #fig.savefig(f"{folder}/frame_{frame_number}.jpg")
        #plt.close()

def draw_all_boxes(box_data_list, idx, color_dict, frame_folder_list, num):
    box_output_folder_list = []
    for i in range(num):
        output_folder = f'temp_box_{i}'
        draw_box(box_data_list[i], idx, color_dict, frame_folder_list[i], output_folder)
        box_output_folder_list.append(output_folder)
    return box_output_folder_list


#### create gif

def merge(box_output_folder_list, map_folder, idx, dpi, num):
    output_folder = 'merge'
    output_folder = make_folder(output_folder)
    if num == 1:
        return merge1(box_output_folder_list[0], map_folder, idx, dpi, output_folder)
    if num == 2:
        return merge2(box_output_folder_list, map_folder, idx, dpi, output_folder)
    if num == 3:
        return merge3(box_output_folder_list, map_folder, idx, dpi, output_folder)

def merge1(box_folder, map_folder, idx, dpi, output_folder):
    
    # Load the two images
    for i in range(idx):
        image1 = cv2.imread(f'{box_folder}/frame_{i}.jpg')
        image2 = cv2.imread(f'{map_folder}/frame_{i}.jpg')
        
        # Resize the images to have the same width
        width = max(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (width, int(image1.shape[0] * width / image1.shape[1])))
        image2 = cv2.resize(image2, (width, int(image2.shape[0] * width / image2.shape[1])))

        # Merge the two images vertically
        merged_image = np.concatenate((image1, image2), axis=0)

        # Save the merged image
        cv2.imwrite(f"{output_folder}/frame_{i}.jpg", merged_image)
        
        '''
        #use plt
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
        '''
    
    return output_folder

def merge2(box_output_folder_list, map_folder, idx, dpi, output_folder):
    box_folder_1, box_folder_2 = box_output_folder_list
    for i in range(idx):
        image_map = cv2.imread(f'{map_folder}/frame_{i}.jpg')
        image1 = cv2.imread(f'{box_folder_1}/frame_{i}.jpg')
        image2 = cv2.imread(f'{box_folder_2}/frame_{i}.jpg')
        
        #height vertical > width horizon
        width, height = image_map.shape[1], image_map.shape[0]
        if height > width:
            img_height = height // 2
            image1 = cv2.resize(image1, (int(image1.shape[1] * img_height / image1.shape[0]), img_height))
            image2 = cv2.resize(image2, (int(image2.shape[1] * img_height / image2.shape[0]), img_height))
            temp = np.concatenate((image1, image2), axis=0) #ver
            temp_height = temp.shape[0]
            image_map = cv2.resize(image_map, (int(image_map.shape[1] * temp_height / image_map.shape[0]), temp_height))
            merged_image = np.concatenate((image_map, temp), axis=1) #hor
        else:
            img_width = width // 2
            image1 = cv2.resize(image1, (img_width, int(image1.shape[0] * img_width / image1.shape[1])))
            image2 = cv2.resize(image2, (img_width, int(image2.shape[0] * img_width / image2.shape[1])))
            temp = np.concatenate((image1, image2), axis=1)
            temp_width = temp.shape[1]
            image_map = cv2.resize(image_map, (temp_width, int(image_map.shape[0] * temp_width / image_map.shape[1])))
            merged_image = np.concatenate((image_map, temp), axis=0)

        # Save the merged image
        cv2.imwrite(f"{output_folder}/frame_{i}.jpg", merged_image)

    return output_folder

def merge3(box_output_folder_list, map_folder, idx, dpi, output_folder):
    box_folder_1, box_folder_2, box_folder_3 = box_output_folder_list
    for i in range(idx):
        image_map = cv2.imread(f'{map_folder}/frame_{i}.jpg')
        image1 = cv2.imread(f'{box_folder_1}/frame_{i}.jpg')
        image2 = cv2.imread(f'{box_folder_2}/frame_{i}.jpg')
        image3 = cv2.imread(f'{box_folder_3}/frame_{i}.jpg')
        
        #height vertical > width horizon
        width, height = image_map.shape[1], image_map.shape[0]
        if height > width:
            img_height = height // 3
            image1 = cv2.resize(image1, (int(image1.shape[1] * img_height / image1.shape[0]), img_height))
            image2 = cv2.resize(image2, (int(image2.shape[1] * img_height / image2.shape[0]), img_height))
            image3 = cv2.resize(image3, (int(image3.shape[1] * img_height / image3.shape[0]), img_height))
            temp = np.concatenate((image1, image2, image3), axis=0) #ver
            temp_height = temp.shape[0]
            image_map = cv2.resize(image_map, (int(image_map.shape[1] * temp_height / image_map.shape[0]), temp_height))
            merged_image = np.concatenate((image_map, temp), axis=1) #hor
        else:
            img_width = width // 3
            image1 = cv2.resize(image1, (img_width, int(image1.shape[0] * img_width / image1.shape[1])))
            image2 = cv2.resize(image2, (img_width, int(image2.shape[0] * img_width / image2.shape[1])))
            image3 = cv2.resize(image3, (img_width, int(image3.shape[0] * img_width / image3.shape[1])))
            temp = np.concatenate((image1, image2, image3), axis=1)
            temp_width = temp.shape[1]
            image_map = cv2.resize(image_map, (temp_width, int(image_map.shape[0] * temp_width / image_map.shape[1])))
            merged_image = np.concatenate((temp, image_map), axis=0)

        # Save the merged image
        cv2.imwrite(f"{output_folder}/frame_{i}.jpg", merged_image)

    return output_folder


#### main

def main(p_m_list, map_path, video_list, box_list, output_format, output_filename, fps, map_folder, dpi, num):
    print('compute homography matrix')
    H_list = map(p_m_list)
    print('convert video to images')
    count, frame_folder_list = get_frames(video_list)
    idx = count - 1
    print('read and sort json file')
    data_list, first_frame_num_list, box_data_list = get_data(box_list, H_list)
    data = merge_map_data(data_list)
    first_frame_num = first_frame_num_list[0]
    print('draw')
    color_dict = color(data)
    print('map')
    draw_map(data, idx, first_frame_num, map_path, color_dict, map_folder)
    print('box')
    box_output_folder_list = draw_all_boxes(box_data_list, idx, color_dict, frame_folder_list, num)
    print('merge')
    folder = merge(box_output_folder_list, map_folder, idx+1, dpi, num)
    if output_format == 'gif':
        print('create gif')
        output_path = output_filename + '.gif'
        create_gif_from_images(output_path, folder, '.png')
    else:
        print('convert images to video')
        output_path = output_filename + '.mp4'
        convert_images_to_video(folder, output_path, fps)
