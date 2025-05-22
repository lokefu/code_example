#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard library imports
import os
import sys
import time
from collections import defaultdict

# Third-party imports
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.morphology import remove_small_objects as skimage_remove_small_objects
import supervision as sv
from supervision.detection.utils import box_iou_batch
from tqdm import tqdm

# Local application/library specific imports
# This is a hack to make this script work from outside the root project folder (without requiring install)
try:
    import lib  # NOQA
except ModuleNotFoundError:
    parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Use abspath for robustness
    if "lib" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        # Try one more level up if __file__ is in a subdirectory of the intended parent
        parent_of_parent_folder = os.path.dirname(parent_folder)
        if "lib" in os.listdir(parent_of_parent_folder):
            sys.path.insert(0, parent_of_parent_folder)
        else:
            raise ImportError(
                f"Can't find path to lib folder! Looked in {parent_folder} and {parent_of_parent_folder}"
            )

from rfdetr import RFDETRBase
from lib.v2_sam.make_sam_v2 import make_samv2_from_original_state_dict
from lib.demo_helpers.video_data_storage import SAM2VideoObjectResults, SimpleSamurai


# --- Configuration Constants ---
VIDEO_PATH = "/home/jupyter/muggled_sam/inputs/others/test_3min_crowd.mp4"
OUTPUT_VIDEO_PATH = "./outputs/cro_sam_hun_refactored.mp4"
SAM_MODEL_PATH = "./model_weights/sam2.1_hiera_base_plus.pt"

DEVICE_PREFERENCE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_PREFERENCE = torch.bfloat16 if DEVICE_PREFERENCE == "cuda" else torch.float32

IMGENC_MAX_SIDE_LENGTH = 1024
IMGENC_USE_SQUARE_SIZING = True

DETECTION_STEP_INTERVAL = 15  # How often to run RFDETR detection and Hungarian matching
SAM_UPDATE_SKIP_FRAMES = 5   # How often to process frames for SAM tracking updates
OCCLUSION_THRESHOLD_FRAMES = 10
SAM_OBJECT_SCORE_THRESHOLD = 2.5
BBOX_IOU_THRESHOLD_MATCHING = 0.5
RSO_MIN_SIZE = 128  # Min size for remove_small_objects
RFDETR_CONFIDENCE_THRESHOLD = 0.5
RFDETR_TARGET_CLASS_ID = 1
# --- End Configuration Constants ---


# --- cucim & CuPy Setup ---
cucim_rso_available = False
cp = None
cucim_rso_func = None

try:
    cp = __import__('cupy')
    from cucim.skimage.morphology import remove_small_objects as _cucim_rso_func
    cucim_rso_func = _cucim_rso_func
    cucim_rso_available = True
    print(f"CuPy (version: {cp.__version__}) and cucim.skimage.morphology.remove_small_objects found. "
          "Will use GPU for remove_small_objects via cucim.")
except ImportError:
    if 'cp' in locals() and cp is not None:
        print(f"CuPy (version: {cp.__version__}) found, but 'cucim.skimage.morphology.remove_small_objects' "
              "is not available. Falling back for RSO.")
    else:
        print("CuPy and/or 'cucim.skimage.morphology.remove_small_objects' not found. "
              "RSO will run on CPU using scikit-image.")
# --- End cucim & CuPy Setup ---


# --- Global State Variables ---
# These are modified by the callback and used across calls
memory_per_obj_dict = defaultdict(SAM2VideoObjectResults.create)
object_index_counter = 0
progress_bar_instance = None
last_annotated_frame_cache = None
# --- End Global State Variables ---


# --- Model Initialization ---
print("Loading SAM model...")
sam_model_config, sam_model_instance = make_samv2_from_original_state_dict(SAM_MODEL_PATH)
sam_model_instance.to(device=DEVICE_PREFERENCE, dtype=DTYPE_PREFERENCE)

print("Loading detector model...")
detector_model_instance = RFDETRBase(device=DEVICE_PREFERENCE)

image_encoding_config = {
    "max_side_length": IMGENC_MAX_SIDE_LENGTH,
    "use_square_sizing": IMGENC_USE_SQUARE_SIZING
}
# --- End Model Initialization ---


# --- Annotators ---
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
# --- End Annotators ---


def normalize_bbox_for_sam_prompt(box_xyxy, frame_width, frame_height):
    """Normalizes a bounding box to [0,1] for SAM prompt."""
    x1, y1, x2, y2 = box_xyxy
    norm_x1 = max(0.0, min(1.0, x1 / frame_width))
    norm_y1 = max(0.0, min(1.0, y1 / frame_height))
    norm_x2 = max(0.0, min(1.0, x2 / frame_width))
    norm_y2 = max(0.0, min(1.0, y2 / frame_height))
    return [(float(norm_x1), float(norm_y1)), (float(norm_x2), float(norm_y2))]

def process_detected_object_mask(binary_mask, track_id_for_display, obj_score):
    """Cleans a binary mask and prepares it for display if score is sufficient."""
    if obj_score < SAM_OBJECT_SCORE_THRESHOLD or not binary_mask.any():
        return None, None

    cleaned_mask = None
    if cucim_rso_available and cp:
        try:
            mask_gpu = cp.asarray(binary_mask)
            cleaned_mask_gpu = cucim_rso_func(mask_gpu, min_size=RSO_MIN_SIZE, connectivity=1)
            cleaned_mask = cp.asnumpy(cleaned_mask_gpu)
        except Exception as e:
            print(f"Error during cucim remove_small_objects: {e}. Falling back to CPU.")
            cleaned_mask = skimage_remove_small_objects(binary_mask, min_size=RSO_MIN_SIZE, connectivity=1)
    else:
        cleaned_mask = skimage_remove_small_objects(binary_mask, min_size=RSO_MIN_SIZE, connectivity=1)

    if cleaned_mask.any():
        return int(track_id_for_display), cleaned_mask
    return None, None


def callback(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Processes a single video frame for object detection and tracking.
    This function is intended to be used as a callback for supervision.process_video.
    """
    global progress_bar_instance, last_annotated_frame_cache, object_index_counter
    global memory_per_obj_dict # Explicitly declare if modifying, defaultdict handles creation

    if progress_bar_instance is None:
        # vcap is a global variable defined outside this function, accessed here.
        # This is okay for scripts but be mindful if refactoring into a class.
        total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) if vcap.isOpened() else 9000
        progress_bar_instance = tqdm(total=total_frames, desc="Processing Video")
    progress_bar_instance.update(1)

    current_frame_height, current_frame_width = frame.shape[:2]

    if frame_idx % SAM_UPDATE_SKIP_FRAMES == 0:
        encoded_imgs_list, _, _ = sam_model_instance.encode_image(frame, **image_encoding_config)
        
        current_frame_display_labels = []
        current_frame_display_masks = []
        tracks_marked_for_deletion = []

        # --- A. Update Existing Tracks ---
        for track_id, obj_memory in list(memory_per_obj_dict.items()):
            obj_score, is_mem_ok, current_mask_tensor, mem_enc, obj_ptr, _ = \
                obj_memory.samurai.step_video_masking(
                    sam_model_instance, encoded_imgs_list, **obj_memory.to_dict()
                )

            if is_mem_ok:
                obj_memory.store_result(frame_idx, mem_enc, obj_ptr)
            else:
                obj_memory.increment_bad_ctr()
                if obj_memory.bad_ctr > OCCLUSION_THRESHOLD_FRAMES:
                    tracks_marked_for_deletion.append(track_id)
                    continue

            obj_mask_resized = torch.nn.functional.interpolate(
                current_mask_tensor, size=(current_frame_height, current_frame_width),
                mode="bilinear", align_corners=False,
            )
            obj_mask_binary = (obj_mask_resized > 0.0).cpu().numpy().squeeze()
            obj_memory.store_mask(obj_mask_binary)

            label_to_display, mask_to_display = process_detected_object_mask(
                obj_mask_binary, track_id, obj_score
            )
            if label_to_display is not None:
                current_frame_display_labels.append(label_to_display)
                current_frame_display_masks.append(mask_to_display)
            elif obj_score < 0 and obj_memory.bad_ctr <= OCCLUSION_THRESHOLD_FRAMES: # Negative score is also bad
                obj_memory.increment_bad_ctr() # Increment again if score is very bad
                if obj_memory.bad_ctr > OCCLUSION_THRESHOLD_FRAMES:
                     if track_id not in tracks_marked_for_deletion:
                        tracks_marked_for_deletion.append(track_id)
        
        for track_id_to_delete in tracks_marked_for_deletion:
            memory_per_obj_dict.pop(track_id_to_delete, None)

        # --- B. New Detections and Hungarian Matching ---
        if frame_idx % DETECTION_STEP_INTERVAL == 0:
            rfdetr_detections = detector_model_instance.predict(frame, threshold=RFDETR_CONFIDENCE_THRESHOLD)
            rfdetr_detections = rfdetr_detections[rfdetr_detections.class_id == RFDETR_TARGET_CLASS_ID]

            new_detection_input_bboxes_xyxy = rfdetr_detections.xyxy
            new_detection_prompts_to_init = []

            if new_detection_input_bboxes_xyxy.shape[0] > 0:
                for i, box_xyxy in enumerate(new_detection_input_bboxes_xyxy):
                    normalized_box_list = normalize_bbox_for_sam_prompt(box_xyxy, current_frame_width, current_frame_height)
                    new_detection_prompts_to_init.append({
                        "original_detection_index": i,
                        "xyxy_for_iou": box_xyxy,
                        "prompts_for_sam_init": {
                            "box_tlbr_norm_list": [normalized_box_list],
                            "fg_xy_norm_list": [], "bg_xy_norm_list": [],
                        }
                    })

            existing_track_bboxes_from_masks = []
            existing_track_ids_for_matching = []
            for track_id, obj_memory in memory_per_obj_dict.items():
                if obj_memory.mask is not None and obj_memory.mask.any():
                    mask_batch = np.expand_dims(obj_memory.mask, axis=0)
                    boxes_from_mask = sv.mask_to_xyxy(mask_batch)
                    if boxes_from_mask.shape[0] > 0:
                        box_from_mask = boxes_from_mask[0]
                        if box_from_mask[0] < box_from_mask[2] and box_from_mask[1] < box_from_mask[3]: # Valid box
                            existing_track_bboxes_from_masks.append(box_from_mask)
                            existing_track_ids_for_matching.append(track_id)
            
            matched_new_detection_indices = set()
            if new_detection_input_bboxes_xyxy.shape[0] > 0 and len(existing_track_bboxes_from_masks) > 0:
                existing_boxes_np = np.array(existing_track_bboxes_from_masks)
                if new_detection_input_bboxes_xyxy.ndim == 2 and existing_boxes_np.ndim == 2 and \
                   new_detection_input_bboxes_xyxy.shape[0] > 0 and existing_boxes_np.shape[0] > 0:
                    
                    iou_matrix = box_iou_batch(new_detection_input_bboxes_xyxy, existing_boxes_np)
                    cost_matrix = 1 - iou_matrix
                    row_indices, col_indices = linear_sum_assignment(cost_matrix)
                    
                    for r_idx, c_idx in zip(row_indices, col_indices):
                        if iou_matrix[r_idx, c_idx] >= BBOX_IOU_THRESHOLD_MATCHING:
                            matched_new_detection_indices.add(new_detection_prompts_to_init[r_idx]["original_detection_index"])
            
            for det_info in new_detection_prompts_to_init:
                if det_info["original_detection_index"] not in matched_new_detection_indices:
                    init_mask_tensor, init_mem, init_ptr = sam_model_instance.initialize_video_masking(
                        encoded_imgs_list, **det_info["prompts_for_sam_init"]
                    )
                    object_index_counter += 1
                    memory_per_obj_dict[object_index_counter].store_prompt_result(frame_idx, init_mem, init_ptr)
                    samurai_tracker = SimpleSamurai(init_mask_tensor)
                    memory_per_obj_dict[object_index_counter].store_samurai(samurai_tracker)
                    # Note: These newly initialized tracks will have their first SAM *update* (step_video_masking)
                    # and display in the *next* frame_idx that is a multiple of skip_frames.
                    # If you need them displayed immediately, you'd have to run a mini-SAM-step here
                    # or add their initial mask (after resizing and cleaning) to current_frame_display_masks.
                    # For simplicity and consistency with the main loop, we let them be processed in the next cycle.

        # --- C. Annotation ---
        annotated_frame = frame.copy() # Start with a fresh copy
        if current_frame_display_masks:
            display_detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(np.array(current_frame_display_masks)),
                mask=np.array(current_frame_display_masks),
                tracker_id=np.array(current_frame_display_labels).astype(int)
            )
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=display_detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=display_detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=display_detections,
                labels=[str(tid) for tid in display_detections.tracker_id]
            )
        
        last_annotated_frame_cache = annotated_frame
        return annotated_frame

    if last_annotated_frame_cache is not None:
        return last_annotated_frame_cache
    return frame.copy() # Fallback for initial frames if nothing cached yet


def main():
    """Main function to set up and run the video processing."""
    # Ensure vcap is opened (it's defined globally)
    if not vcap.isOpened():
        print(f"Re-opening video capture for: {VIDEO_PATH}")
        vcap.open(VIDEO_PATH) # cv2.VideoCapture(VIDEO_PATH)
        if not vcap.isOpened():
            raise IOError(f"Error: Could not open video: {VIDEO_PATH}")
        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset if re-opened

    process_start_time = time.time()

    sv.process_video(
        source_path=VIDEO_PATH,
        target_path=OUTPUT_VIDEO_PATH,
        callback=callback
    )

    if progress_bar_instance:
        progress_bar_instance.close()

    if vcap.isOpened():
        vcap.release()

    total_processing_time = time.time() - process_start_time
    print(f"\nTotal processing time: {total_processing_time:.2f} seconds")
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    # Initialize video capture here to make 'vcap' available globally for the callback's progress bar
    vcap = cv2.VideoCapture(VIDEO_PATH)
    if not vcap.isOpened():
        raise IOError(f"Unable to read video frames from: {VIDEO_PATH} at script start.")
    vcap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    ok_frame, _ = vcap.read() # Read one frame to confirm
    if not ok_frame:
        vcap.release()
        raise IOError(f"Unable to read first frame from: {VIDEO_PATH} at script start.")
    vcap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start for sv.process_video

    main()