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
  
#### get file list in a folder
# extract number in file name
import re

def extract_frame_number(file_name):
    # Extract the frame number from the image file name, frame_00001.jpg -> 1
    return int(re.search(r'\d+', file_name).group())

def extract_alpha(file_name):
    return ''.join(filter(str.isalpha, file_name))

def get_file_list(input_folder, sort_rule, file_type1, file_type2=None):
    # Get the list of image files in the input folder
    if sort_rule == 'ending_number':
        return sorted([f for f in os.listdir(input_folder) if f.endswith(file_type1) or f.endswith(file_type2)], key=extract_frame_number)
    elif sort_rule == 'alpha':
        return sorted([f for f in os.listdir(input_folder) if f.endswith(file_type1) or f.endswith(file_type2)], key=extract_alpha)


