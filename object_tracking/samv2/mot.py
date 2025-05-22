#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    import os
    import sys

    parent_folder = os.path.dirname(os.path.dirname(__file__))
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        raise ImportError("Can't find path to lib folder!")

import time
from tqdm import tqdm
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv

import torch
from rfdetr import RFDETRBase
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults, SimpleSamurai
from skimage.morphology import remove_small_objects as skimage_remove_small_objects # Fallback


# --- cucim & CuPy Setup ---
cucim_rso_available = False
cp = None  # Will hold the cupy module if import is successful
cucim_rso_func = None # Will hold the cucim remove_small_objects function

try:
    cp = __import__('cupy') # Import cupy
    from cucim.skimage.morphology import remove_small_objects as _cucim_rso_func
    cucim_rso_func = _cucim_rso_func
    cucim_rso_available = True
    print(f"CuPy (version: {cp.__version__}) and cucim.skimage.morphology.remove_small_objects found. Will use GPU for remove_small_objects via cucim.")
except ImportError:
    if 'cp' in locals() and cp is not None:
        print(f"CuPy (version: {cp.__version__}) found, but 'cucim.skimage.morphology.remove_small_objects' is not available. Falling back for RSO.")
    else:
        print("CuPy and/or 'cucim.skimage.morphology.remove_small_objects' not found. RSO will run on CPU using scikit-image.")
# --- End cucim & CuPy Setup ---


# Define pathing & device usage
video_path = "/home/jupyter/muggled_sam/inputs/others/IMG_1240.MOV"
output_path = "./outputs/IMG_1240_sam.mp4"
model_path = "./model_weights/sam2.1_hiera_base_plus.pt"
device, dtype = "cpu", torch.float32
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16

# Define image processing config (shared for all video frames)
imgenc_config_dict = {"max_side_length": 1024, "use_square_sizing": True}

# Set up memory storage for tracked objects
# -> Assumes each object is represented by a unique dictionary key (e.g. 'obj1')
# -> This holds both the 'prompt' & 'recent' memory data needed for tracking!
memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)

# Read first frame to check that we can read from the video, then reset playback
vcap = cv2.VideoCapture(video_path)
vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # See: https://github.com/opencv/opencv/issues/26795
ok_frame, first_frame = vcap.read()
if not ok_frame:
    raise IOError(f"Unable to read video frames: {video_path}")
vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Set up model
print("Loading model...")
model_config_dict, sammodel = make_samv2_from_original_state_dict(model_path)
sammodel.to(device=device, dtype=dtype)
model = RFDETRBase(device=device)

# How often to redo detection
step = 15
skip_frames = 5
# How many tracking frames it can be occluded before it gets removed
# E.g threshold of 10 at skip_frames of 5 => 10/(15/5) 3.33s before its removed after complete occlusion
# Increasing this threshold increases processing time
occlusion_threshold = 10
# At what SAM2 object score to remove for display to reduce artifacts
obj_score_threshold = 2.5
# IOU Threshold
iou_threshold = 0.5
# object tracker index
object_index = 0
# tqdm progress bar
progress_bar = None

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)

def get_existing_mask(frame, mask):
    non_none_masks = [object_memory.mask for _, object_memory in memory_per_obj_dict.items() if object_memory.mask is not None]

    if len(non_none_masks) > 0:
        obj_mask = torch.nn.functional.interpolate(
            mask,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()

        iou_scores = sv.mask_iou_batch(np.array(non_none_masks), np.array([obj_mask_binary]))

        if np.any(iou_scores > iou_threshold):
            return True
    
    return None

def callback(frame, frame_idx):
    # tqdm progress bar
    global progress_bar
    if progress_bar is None:
        progress_bar = tqdm(total=900, desc="Processing Video")
    progress_bar.update(1)

    if (frame_idx % skip_frames == 0):
        # Encode frame data (shared for all objects)
        encoded_imgs_list, _, _ = sammodel.encode_image(frame, **imgenc_config_dict)

        # Storage for results
        label_result = []
        mask_result = []
        
        marked_for_deletion = []
        # Update tracking using newest frame
        for obj_key_name, obj_memory in memory_per_obj_dict.items():
            # 1. step_video_masking
            obj_score, is_mem_ok, best_mask_pred, mem_enc, obj_ptr, xy1xy2_kal = obj_memory.samurai.step_video_masking(
                sammodel, encoded_imgs_list, **obj_memory.to_dict()
            )

            # Store memory if samurai says its ok for tracking
            if is_mem_ok:
                # Store 'recent' memory encodings from current frame (helps track objects with changing appearance)
                # -> This can be commented out and tracking may still work, if object doesn't change much
                obj_memory.store_result(frame_idx, mem_enc, obj_ptr)
            else:
                obj_memory.increment_bad_ctr()
                if obj_memory.bad_ctr > occlusion_threshold:
                    marked_for_deletion.append(obj_key_name)
                    continue

            # Add object mask prediction to 'combine' mask for display
            # -> This is just for visualization, not needed for tracking
            obj_mask = torch.nn.functional.interpolate(
                best_mask_pred,
                size=frame.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()

            obj_memory.store_mask(obj_mask_binary)

            # But skip it for display if score is bad to reduce artifacts.
            if obj_score < obj_score_threshold:
                continue

            if cucim_rso_available and cp is not None:
                try:
                    obj_mask_gpu = cp.asarray(obj_mask_binary)
                    cleaned_mask_gpu = cucim_rso_func(obj_mask_gpu, min_size=128, connectivity=1)
                    obj_mask_binary = cp.asnumpy(cleaned_mask_gpu)
                except Exception:
                    obj_mask_binary = skimage_remove_small_objects(obj_mask_binary, min_size=128, connectivity=1)
            else:
                obj_mask_binary = skimage_remove_small_objects(obj_mask_binary, min_size=128, connectivity=1)
            
            label_result.append(int(obj_key_name))
            mask_result.append(obj_mask_binary)
            
        for to_delete in marked_for_deletion:
            memory_per_obj_dict.pop(to_delete, None)

        # Doing detection after the mask are updated for this frame improves results
        frame_prompts_dict = {}
        if (frame_idx % step == 0):
            # Generate prompt dict from detection model
            detections = model.predict(frame, threshold=0.5)
            detections = detections[detections.class_id == 1]

            image_height, image_width = frame.shape[:2]
            for i, box in enumerate(detections.xyxy):
                x1, y1, x2, y2 = box
                norm_x1 = x1 / image_width
                norm_y1 = y1 / image_height
                norm_x2 = x2 / image_width
                norm_y2 = y2 / image_height

                # Clamp values to [0.0, 1.0] to handle potential rounding errors
                # or boxes slightly outside the frame.
                norm_x1 = max(0.0, min(1.0, norm_x1))
                norm_y1 = max(0.0, min(1.0, norm_y1))
                norm_x2 = max(0.0, min(1.0, norm_x2))
                norm_y2 = max(0.0, min(1.0, norm_y2))
                formatted_box_list = [(float(norm_x1), float(norm_y1)), (float(norm_x2), float(norm_y2))]

                frame_prompts_dict[i] = {
                    "box_tlbr_norm_list": [formatted_box_list],
                    "fg_xy_norm_list": [],
                    "bg_xy_norm_list": [],
                }
        
        # Moved prompt generation to after mask generation for better iou as you are not comparing against the previous frame
        # Generate & store prompt memory encodings for each object as needed
        prompts_dict = frame_prompts_dict if frame_prompts_dict else None
        if prompts_dict is not None:

            # Loop over all sets of prompts for the current frame
            for obj_key_name, obj_prompts in prompts_dict.items():
                init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(encoded_imgs_list, **obj_prompts)
                    
                existing_masks = get_existing_mask(frame, init_mask)

                if existing_masks is None:
                    global object_index
                    object_index = (object_index or 0) + 1
                    memory_per_obj_dict[object_index].store_prompt_result(frame_idx, init_mem, init_ptr)
                    samurai = SimpleSamurai(init_mask)
                    memory_per_obj_dict[object_index].store_samurai(samurai)

                # Directly show mask for first frame
                if frame_idx == 0:
                    obj_mask = torch.nn.functional.interpolate(
                        init_mask,
                        size=frame.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    )
                    obj_mask_binary = (obj_mask > 0.0).cpu().numpy().squeeze()
                    obj_mask_gpu = cp.asarray(obj_mask_binary)
                    cleaned_mask_gpu = cucim_rso_func(obj_mask_gpu, min_size=128, connectivity=1)
                    obj_mask_binary = cp.asnumpy(cleaned_mask_gpu)
                    label_result.append(int(object_index))
                    mask_result.append(obj_mask_binary)
       
        global stored_annotated_frame
        
        if len(mask_result) == 0:
            stored_annotated_frame = frame
            return frame

        # Write results to frame
        mask_detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(np.array(mask_result)),
            mask=np.array(mask_result),
            tracker_id=np.array(label_result)
        )
        annotated_frame = mask_annotator.annotate(
            scene=frame.copy(),
            detections=mask_detections,
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=mask_detections,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=mask_detections,
            labels=[str(i) for i in label_result]
        )

        stored_annotated_frame = annotated_frame

    return stored_annotated_frame

process_time_start = time.time()

sv.process_video(
    source_path=video_path,
    target_path=output_path,
    callback=callback
)
if progress_bar:
    progress_bar.close()

process_time = time.time() - process_time_start 
print('process_time', process_time)