#------- libraries -------#
import os
import time
import json
import random
from tqdm import tqdm
from os import path
from datetime import datetime

import cv2
import numpy as np
import supervision as sv

import torch
import torch.nn as nn
from models import build_model
from models.utils import load_checkpoint
from models.runtime_tracker import RuntimeTracker, TI, TS
from utils.utils import distributed_rank


#------- input & output -------#
video_folders_nested = False
# Specify whether the video folders are nested, i.e.,
'''
-raw
    -March 1
        -video1.mp4
    -March 2
'''
# Replace '/path/to/your/directory' with the actual path you want to inspect
#video_folders_path = '/mnt/iora/batch-1/raw/' #this is nested
video_folders_path = '/mnt/iora/batch-1/raw/March 2/'

json_folder_path = '/mnt/iora/batch-1/processed/'

output_folder = '/home/jupyter/iora_putr_output/'

#output folder, format like
'''
-iora_putr_output
    -json
        -video1.json
    -vis
        -video1.mp4
'''

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# create subfolders for json and vis
if not os.path.exists(os.path.join(output_folder, 'json/')):
    os.makedirs(os.path.join(output_folder, 'json/'))
if not os.path.exists(os.path.join(output_folder, 'vis/')):
    os.makedirs(os.path.join(output_folder, 'vis/'))

# write log
output_log_path = output_folder + "log.txt"
# write log into json
output_log_json = output_folder + 'log.json'
global log_dict
log_dict = {}

#------- variables and parameters -------#
random.seed(0)

config = {'GIT_VERSION': None, 'MODE': 'submit', 'CONFIG_PATH': './configs/eval_dancetrack_putr.yaml', 'VISUALIZE': True, 
          'AVAILABLE_GPUS': '0,', 'DEVICE': 'cuda', 'OUTPUTS_DIR': './outputs/', 'USE_DISTRIBUTED': False, 
          'SUBMIT_MODEL': 'dancetrack.pth', 'SUBMIT_DATA_SPLIT': 'val', 'DET_SCORE_THRESH': 0.1, 'TRACK_SCORE_THRESH': 0.6, 
          'MISS_TOLERANCE': 30, 'MAX_NFRAMES': 30, 'MIN_TRACK_HITS': 3, 'NEW_TRK_TOLERANCE': 3, 'DETS_IOU_THRESH': 0.3, 'ASSO_THRE1': 0.2, 
          'ASSO_THRE2': 0.2, 'DIM': 512, 'N_LAYERS': 6, 'N_HEADS': 8, 'NORM_EPS': 1e-05, 'PATCH_GRID': 64, 'MAX_SEQ_LEN': 4096, 'DATASET': 'DanceTrack', 'DATA_ROOT': 'datasets2'}
#print('miss tolerance: ', config['MISS_TOLERANCE'])


#------- putr -------#
class Submitter:
    def __init__(self, config, dataset_name: str, outputs_dir: str, model: nn.Module, all_detections: dict):
        self.dataset_name = dataset_name
        self.outputs_dir = outputs_dir
        self.model = model
        
        self.tracker = RuntimeTracker(config, model)
        self.device = next(self.model.parameters()).device

        self.visualize = config["VISUALIZE"]
        self.model.eval()
        
        # Initialize supervision components for visualization
        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        
        self.min_track_hits = config["MIN_TRACK_HITS"]
        self.detection = all_detections
        return

    @torch.no_grad()
    def run(self):
        time_per_frame = []
        time_per_frame_with_vis = []
        video_path = self.dataset_name
        cap = cv2.VideoCapture(video_path)
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Create output video file if visualization is enabled
        if self.visualize:
            output_video_path = os.path.join(self.outputs_dir, os.path.basename(video_path))
            output_video = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )

        i = 0
        frame_to_tracks = {}  # Dictionary to store frame_number: "trk_ids: [xyxy]"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count for progress bar

        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if str(i) not in self.detection:
                    i += 1
                    pbar.update(1)
                    continue

                start_time = time.time()

                # load the detections
                detections = self.detection[str(i)]
                frame_torch = torch.from_numpy(frame).unsqueeze(0).to(self.device)
                
                det_bboxes = np.array(detections['boxes']).astype(np.float32)
                det_scores = np.array(detections['scores']).astype(np.float32)
                det_labels = np.ones(len(detections['scores']), dtype=int)
                
                if det_labels.size == 0:
                    i += 1
                    pbar.update(1)
                    continue
                
                #det_bboxes = result.xyxy
                #det_scores = result.confidence
                #det_labels = result.class_id
                #print(i)
                #print(det_bboxes)
                #print(det_scores)
                #print(det_labels)
                det_bboxes = torch.cat([torch.tensor(det_bboxes, device=self.device),
                                        torch.tensor(det_scores, device=self.device).unsqueeze(1),
                                        torch.tensor(det_labels, device=self.device).unsqueeze(1)],
                                        dim=1)
                det_bboxes = det_bboxes.to(self.device)

                # Process detections and tracking
                trks = self.tracker.update(frame_torch, det_bboxes, det_bboxes)
                trks = trks.cpu().numpy()

                dets = trks[:, TI.TLBR:TI.TLBR + 4].astype("int32")
                trk_ids = trks[:, TI.TrackID].astype("int32")

                # Add to frame_to_tracks dictionary
                frame_to_tracks[i] = {trk_id: det.tolist() for trk_id, det in zip(trk_ids, dets)}
                
                end_time = time.time()
                time_per_frame.append(end_time - start_time)
                
                # Visualization using supervision
                if self.visualize:
                    # Create detections in supervision format
                    sv_detections = sv.Detections(
                        xyxy=dets,
                        tracker_id=trk_ids
                    )
                    
                    # Annotate the frame with tracking information
                    annotated_frame = self.box_annotator.annotate(
                        scene=frame.copy(),
                        detections=sv_detections,
                    )
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame,
                        detections=sv_detections,
                        labels=[str(i) for i in trk_ids]
                    )
                    
                    # Write the annotated frame to the output video
                    output_video.write(annotated_frame)

                i += 1

                # Track runtime per frame
                end_time_with_vis = time.time()
                time_per_frame_with_vis.append(end_time_with_vis - start_time)
                # Update progress bar
                pbar.update(1)

        cap.release()
        if self.visualize and 'output_video' in locals():
            output_video.release()

        # Print runtime statistics
        avg_runtime = np.nanmean(time_per_frame)
        avg_runtime_with_vis = np.nanmean(time_per_frame_with_vis)
        print(f"Average Runtime per Frame: {avg_runtime:.4f} seconds")
        print(f"Average Runtime per Frame with visulization: {avg_runtime_with_vis:.4f} seconds")
        print(f"Average FPS: {1 / avg_runtime:.2f}")
        print(f"Average FPS with visulization: {1 / avg_runtime_with_vis:.2f} seconds")

        # write log
        with open(output_log_path, 'a') as f:
            # Use the .write() method to write the string to the file
            result_string = f"Average Runtime per Frame: {avg_runtime:.4f} seconds\n"
            result_string += f"Average Runtime per Frame with visulization: {avg_runtime_with_vis:.4f} seconds\n"
            result_string += f"Average FPS: {1 / avg_runtime:.2f}\n"
            result_string += f"Average FPS with visulization: {1 / avg_runtime_with_vis:.2f} seconds\n"
            result_string += f"\n"
            # Write the string to the file
            f.write(result_string)

        print(f"Successfully wrote results to {output_log_path}")
        
        global log_dict
        # Save the log dictionary to a dict
        log_dict[video_path] = {
            "spf": avg_runtime,
            "spf_vis": avg_runtime_with_vis,
            "fps": 1 / avg_runtime,
            "fps_vis": 1 / avg_runtime_with_vis,
        }

        return frame_to_tracks


os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if config["USE_DISTRIBUTED"]:
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(distributed_rank())

model = build_model(config=config)
load_checkpoint(
    model=model,
    path=config["SUBMIT_MODEL"]
)

#### save into json
def save(path, data):    
    # Save the dictionary to a JSON file
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

#### load from json
def load(path):
    # Load the JSON file
    with open(path, 'r') as json_file:
        return json.load(json_file)

def all_combine_batch(video_path, all_detections, output_dir):
    #all_detections = load(all_detections_path)
    vis_output_dir = os.path.join(output_dir, 'vis/')
    submitter = Submitter(
        config=config,
        dataset_name=video_path,
        outputs_dir=vis_output_dir,
        model=model,
        all_detections=all_detections,
    )
    track_list = submitter.run()
    output = convert_output(track_list)
    json_output_dir = os.path.join(output_dir, 'json/')
    output_tracking_path = os.path.join(json_output_dir, os.path.basename(all_detections_path))
    save(output_tracking_path, output)
    return output


#------- util -------#
def convert_output(track_list):
    # Dictionary to store the transformed data
    output_data = {}

    # Iterate through each frame and its tracking data
    for frame, track_data_in_frame in track_list.items():
        # Iterate through each tracked object (ID and box) in the current frame
        for object_id, bounding_box in track_data_in_frame.items():
            # Check if this object_id is already in our output structure
            
            if str(object_id) not in output_data:
                # If not, initialize its entry with empty lists for frame and box
                object_id_str = str(object_id)
                output_data[object_id_str] = {"frames": [], "boxes": []}

            # Append the current frame number and bounding box to the respective lists
            # for this object ID
            object_id_str = str(object_id)
            output_data[object_id_str]["frames"].append(frame)
            output_data[object_id_str]["boxes"].append(bounding_box)

    return output_data

#get video and json path
def get_folder_names(path):
  """
  Gets the names of all folders (directories) directly inside a given path.

  Args:
    path (str): The path to the directory to list.

  Returns:
    list: A list of strings, where each string is the name of a folder
          found directly inside the given path. Returns an empty list
          if the path does not exist, is not a directory, or contains
          no subdirectories.
  """
  folder_names = []
  
  # Check if the provided path exists and is a directory
  if not os.path.isdir(path):
    print(f"Error: Path '{path}' does not exist or is not a directory.")
    return []

  # List all entries (files and directories) in the path
  entries = os.listdir(path)

  # Iterate through the entries
  for entry_name in entries:
    # Construct the full path for the entry
    full_path = os.path.join(path, entry_name)
    
    # Check if the entry is a directory
    if os.path.isdir(full_path):
      # If it's a directory, add its name to our list
      folder_names.append(entry_name)
      
  return folder_names

def get_files_with_ending_in_folder(path, ending):
  """
  Gets the names of all files ending with .mp4 directly inside a given path.

  Args:
    path (str): The path to the directory to search within.

  Returns:
    list: A list of strings, where each string is the name of an .mp4 file
          found directly inside the given path. Returns an empty list
          if the path does not exist, is not a directory, or contains
          no .mp4 files directly within it.
  """
  files = []
  target_extension = ending

  # Check if the provided path exists and is a directory
  if not os.path.isdir(path):
    print(f"Error: Path '{path}' does not exist or is not a directory.")
    return []

  # List all entries (files and directories) in the path
  entries = os.listdir(path)

  # Iterate through the entries
  for entry_name in entries:
    # Construct the full path for the entry
    full_path = os.path.join(path, entry_name)

    # Check if the entry is a file AND if its name ends with the target extension
    # We use .lower() to handle both .mp4 and .MP4 case extensions
    if os.path.isfile(full_path) and entry_name.lower().endswith(target_extension):
      # If it's a file ending with .mp4, add its name to our list
      files.append(entry_name)

  return files


#------- prepare input -------#
# Check if the folders are nested
if video_folders_nested:
    video_folder_list = get_folder_names(video_folders_path)
    video_folder_list_path = [video_folders_path + video_folder + '/' for video_folder in video_folder_list]

    video_list = []
    for video_folder in video_folder_list_path:
        video_files = get_files_with_ending_in_folder(video_folder, '.mp4')
        for video_file in video_files:  
            video_file_path = video_folder + video_file
            video_list.append([video_file_path, video_file])
else:
    video_list = []
    video_files = get_files_with_ending_in_folder(video_folders_path, '.mp4')
    for video_file in video_files:  
        video_file_path = video_folders_path + video_file
        video_list.append([video_file_path, video_file])
    print(video_list)

video_json_pair_list = []
for video_file in video_list:
    video_file_path = video_file[0]
    video_file_name = video_file[1]
    json_file_name = video_file_name.replace('.mp4', f'.json')
    json_file_path = os.path.join(json_folder_path, json_file_name)
    if os.path.exists(json_file_path):
        video_json_pair_list.append([video_file_path, json_file_path])
    else:
        print(f"Warning: JSON file '{json_file_path}' does not exist for video '{video_file_path}'. Skipping this video.")
        continue


#------- run batch -------#
for video_json_pair in video_json_pair_list:
    video_path = video_json_pair[0]
    all_detections_path = video_json_pair[1]
    print(video_path)
    print(all_detections_path)
    
    # write log
    with open(output_log_path, 'a') as f:
        # Use the .write() method to write the string to the file
        result_string = f"Processing video: {video_path}\n"
        result_string += f"Processing json: {all_detections_path}\n"
        result_string += f"\n"
        # Write the string to the file
        f.write(result_string)
    
    # feed detections only into the model
    all_detections = load(all_detections_path)
    detections = all_detections['frames']
    output = all_combine_batch(video_path, detections, output_folder)

save(output_log_json, log_dict)
print(f"Successfully wrote results to {output_log_json}")