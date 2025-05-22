#### save into json
import json
def save(path, data):    
    # Save the dictionary to a JSON file
    with open(path, 'w') as json_file:
        json.dump(data, json_file)

#### load from json
import json
def load(path):
    # Load the JSON file
    with open(path, 'r') as json_file:
        return json.load(json_file)


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


random.seed(0)

all_detections_path = '/home/jupyter/test/PuTR/output/Copy of A iORA Isetan CHANNEL  3  (1100-2330)      1 MAR 2025_segment_1.json'
all_detections = load(all_detections_path)
detections = all_detections#['frames']
#save('frames_only_Copy of A iORA Isetan CHANNEL  3  (1100-2330)      1 MAR 2025.json', all_detections['frames'])

video_path = '/home/jupyter/test/PuTR/output/Copy of A iORA Isetan CHANNEL  3  (1100-2330)      1 MAR 2025_segment_1.mp4'


config = {'GIT_VERSION': None, 'MODE': 'submit', 'CONFIG_PATH': './configs/eval_dancetrack_putr.yaml', 'VISUALIZE': True, 
          'AVAILABLE_GPUS': '0,', 'DEVICE': 'cuda', 'OUTPUTS_DIR': './outputs/', 'USE_DISTRIBUTED': False, 
          'SUBMIT_MODEL': 'dancetrack.pth', 'SUBMIT_DATA_SPLIT': 'val', 'DET_SCORE_THRESH': 0.1, 'TRACK_SCORE_THRESH': 0.6, 
          'MISS_TOLERANCE': 10, 'MAX_NFRAMES': 30, 'MIN_TRACK_HITS': 3, 'NEW_TRK_TOLERANCE': 3, 'DETS_IOU_THRESH': 0.3, 'ASSO_THRE1': 0.2, 
          'ASSO_THRE2': 0.2, 'DIM': 512, 'N_LAYERS': 6, 'N_HEADS': 8, 'NORM_EPS': 1e-05, 'PATCH_GRID': 64, 'MAX_SEQ_LEN': 4096, 'DATASET': 'DanceTrack', 'DATA_ROOT': 'datasets2'}
print('miss tolerance: ', config['MISS_TOLERANCE'])
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
        print(f"Average FPS with visulization: {1 / avg_runtime_with_vis:.2f}")

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

def all_combine(video_path, detections, output_dir):
    #all_detections = load(all_detections_path)
    submitter = Submitter(
        config=config,
        dataset_name=video_path,
        outputs_dir=output_dir,
        model=model,
        all_detections=detections,
    )
    track_list = submitter.run()
    output = convert_output(track_list)
    output_tracking_path = os.path.join(output_dir, os.path.basename(all_detections_path))
    save(output_tracking_path, output)
    return output


output = all_combine(video_path, detections, 'output_test_10/')