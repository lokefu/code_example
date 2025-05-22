import io
import requests
# import supervision as sv # No longer needed for drawing
from PIL import Image
import cv2
import time
import numpy as np
import math # For ceiling function
import traceback # For detailed error printing
import onnxruntime as ort # Import ONNX Runtime
import sys # For exiting
import threading # For running inference in background
import queue # Import the queue module itself
import os # Needed for checking cache path

# --- Configuration ---
# *** IMPORTANT: Set your RTSP stream URL here ***
RTSP_URL = "rtsp://your_stream_url_here"
# *** Path to the ORIGINAL FP32 ONNX model ***
ONNX_MODEL_PATH = 'model/inference_model.sim.onnx' # <<< CHANGED BACK TO FP32 MODEL
# *** IMPORTANT: Set a valid, writable path for the TensorRT engine cache ***
TENSORRT_CACHE_PATH = '/path/to/your/trt_cache' # e.g., '/home/jetson/my_app_trt_cache/'
# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5
# Region of Interest (ROI) vertical boundaries (percentage of frame height)
ROI_TOP_PERCENT = 0.50
ROI_BOTTOM_PERCENT = 0.85
# --- ONNX Model Input Configuration ---
ONNX_INPUT_SIZE = (560, 560) # Example: (height, width)
# Normalization parameters
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# --- Drawing Configuration ---
BOX_COLOR = (0, 0, 255) # BGR format for OpenCV - Red
BOX_THICKNESS = 2

# --- Global variables for thread communication ---
# Use LifoQueue size 1: Inference always gets the latest frame, main thread always gets latest results
frame_queue = queue.LifoQueue(maxsize=1)
results_queue = queue.LifoQueue(maxsize=1) # Will store numpy arrays of boxes now
# Flag to signal threads to stop
stop_event = threading.Event()


# --- ONNX Model Initialization ---
print(f"Attempting to load ONNX model from: {ONNX_MODEL_PATH}")
session = None
input_name = None
output_names = None
try:
    available_providers = ort.get_available_providers()
    print(f"Available ONNX Runtime providers: {available_providers}")

    # --- Provider Selection for Jetson ---
    preferred_providers = []
    provider_options = []

    # Check and add TensorRT
    if 'TensorrtExecutionProvider' in available_providers:
        print("TensorRT Execution Provider available.")
        preferred_providers.append('TensorrtExecutionProvider')

        # --- Enable TensorRT Caching ---
        print(f"Checking TensorRT cache path: {TENSORRT_CACHE_PATH}")
        if not os.path.exists(TENSORRT_CACHE_PATH):
            print(f"Cache directory not found. Creating: {TENSORRT_CACHE_PATH}")
            try:
                os.makedirs(TENSORRT_CACHE_PATH)
                print("Cache directory created.")
            except OSError as e:
                print(f"WARNING: Could not create TensorRT cache directory: {e}. Caching will be disabled.")
                TENSORRT_CACHE_PATH = None # Disable caching if path fails

        if TENSORRT_CACHE_PATH and os.access(TENSORRT_CACHE_PATH, os.W_OK):
             print("TensorRT engine caching enabled.")
             trt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 2147483648, # 2GB
                'trt_fp16_enable': False, # <<< CHANGED TO FALSE FOR FP32
                'trt_engine_cache_enable': True, # Enable caching
                'trt_engine_cache_path': TENSORRT_CACHE_PATH, # Use the verified path
             }
        else:
            print("TensorRT engine caching disabled (path not found or not writable).")
            trt_options = { # Options without caching
                'device_id': 0,
                'trt_max_workspace_size': 2147483648,
                'trt_fp16_enable': False, # <<< CHANGED TO FALSE FOR FP32
            }
        provider_options.append(trt_options)
        # ------------------------------

    else:
        print("TensorRT Execution Provider *not* available.")

    # Check and add CUDA
    if 'CUDAExecutionProvider' in available_providers:
        print("CUDA Execution Provider available.")
        preferred_providers.append('CUDAExecutionProvider')
        provider_options.append({}) # Empty dict to match length
    else:
        print("CUDA Execution Provider *not* available.")

    # Always add CPU as fallback
    preferred_providers.append('CPUExecutionProvider')
    provider_options.append({}) # Empty dict for CPU options

    print(f"Attempting to load model with providers: {preferred_providers}")
    if len(provider_options) != len(preferred_providers):
         print("WARNING: Provider options length mismatch. Resetting options.")
         provider_options = None

    # --- Load Session ---
    print("Creating ONNX Runtime session...") # DEBUG PRINT
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=preferred_providers,
        provider_options=provider_options
    )
    print("ONNX Runtime session created successfully.") # DEBUG PRINT

    chosen_provider = session.get_providers()
    print(f"ONNX Runtime session using provider(s): {chosen_provider}")
    if 'TensorrtExecutionProvider' not in chosen_provider and 'CUDAExecutionProvider' not in chosen_provider:
        print("Warning: GPU acceleration (TensorRT/CUDA) is not being used.")
        print("Specifically check CUDA/cuDNN library paths and versions required by your ONNX Runtime build.")

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model Input Name: {input_name}")
    print(f"Model Output Names: {output_names}")
    input_type = session.get_inputs()[0].type
    print(f"Model Expected Input Type: {input_type}") # Should be tensor(float)
    print("--- ONNX Initialization Complete ---") # DEBUG PRINT

except FileNotFoundError:
    print(f"ERROR: ONNX model not found at '{ONNX_MODEL_PATH}'")
    sys.exit(1)
except Exception as e:
    print(f"Error loading ONNX model or creating session: {e}")
    print("Ensure the correct ONNX Runtime for Jetson is installed and model path is valid.")
    print("If using TensorRT/CUDA, check provider options and CUDA/cuDNN/TensorRT library installations and paths (LD_LIBRARY_PATH).")
    traceback.print_exc()
    sys.exit(1)

# --- Preprocessing Function ---
# (Remains the same - returns float32)
def preprocess_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """ Preprocesses image for ONNX model, returns np.float32 """
    h, w = target_size
    resized_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    float_image = rgb_image.astype(np.float32) / 255.0
    normalized_image = (float_image - NORM_MEAN) / NORM_STD
    chw_image = normalized_image.transpose(2, 0, 1)
    batch_image = np.expand_dims(chw_image, axis=0)
    return batch_image.astype(np.float32)

# --- Postprocessing Function ---
def postprocess_output(outputs: list, confidence_threshold: float, original_shape: tuple) -> np.ndarray | None:
    """
    Postprocesses ONNX output (expects FP32), returns NumPy array of boxes [x1, y1, x2, y2]
    relative to ROI, or None if no detections pass threshold.
    """
    if len(outputs) != 2:
         return None

    pred_boxes_raw = outputs[0]
    pred_labels_scores_raw = outputs[1]

    # Ensure outputs are float32 (should be, but safe check)
    if pred_boxes_raw.dtype != np.float32:
        pred_boxes_raw = pred_boxes_raw.astype(np.float32)
    if pred_labels_scores_raw.dtype != np.float32:
        pred_labels_scores_raw = pred_labels_scores_raw.astype(np.float32)

    if pred_boxes_raw.ndim != 3 or pred_labels_scores_raw.ndim != 3 or pred_boxes_raw.shape[0] != 1 or pred_labels_scores_raw.shape[0] != 1:
        return None

    pred_boxes = pred_boxes_raw[0]
    pred_labels_scores = pred_labels_scores_raw[0]

    if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4 or pred_labels_scores.ndim != 2 or pred_labels_scores.shape[1] != 2:
        return None

    try:
        scores = pred_labels_scores[:, 1]
    except IndexError:
        return None

    keep = (scores > confidence_threshold)
    # scores_filtered = scores[keep] # No longer needed unless displaying score
    boxes_filtered = pred_boxes[keep]

    if len(boxes_filtered) == 0:
        return None

    # Convert boxes (assuming NORMALIZED cx, cy, w, h)
    try:
        roi_h, roi_w = original_shape
        cx_norm, cy_norm, w_norm, h_norm = boxes_filtered.T

        cx = cx_norm * roi_w
        cy = cy_norm * roi_h
        w = w_norm * roi_w
        h = h_norm * roi_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        xyxy_roi = np.stack([x1, y1, x2, y2], axis=1)

        # Clip coordinates to ROI boundaries
        xyxy_roi[:, 0] = np.clip(xyxy_roi[:, 0], 0, roi_w - 1)
        xyxy_roi[:, 1] = np.clip(xyxy_roi[:, 1], 0, roi_h - 1)
        xyxy_roi[:, 2] = np.clip(xyxy_roi[:, 2], 0, roi_w - 1)
        xyxy_roi[:, 3] = np.clip(xyxy_roi[:, 3], 0, roi_h - 1)

    except Exception as e:
         print(f"ERROR during box conversion/scaling: {e}")
         traceback.print_exc()
         return None

    # Return the numpy array of ROI-relative boxes
    return xyxy_roi


# --- Inference Function (for the worker thread) ---
def inference_worker():
    """
    Worker thread function to perform inference on frames from the queue.
    Puts final adjusted xyxy coordinates (NumPy array or None) into results queue.
    """
    print("Inference worker thread started.")
    is_first_inference = True # Flag for first inference timing with TRT
    while not stop_event.is_set():
        try:
            frame_to_process = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        worker_start_time = time.time()
        adjusted_boxes_result = None # Default to None
        try:
            height, width, _ = frame_to_process.shape
            roi_start_row = int(height * ROI_TOP_PERCENT)
            roi_end_row = int(height * ROI_BOTTOM_PERCENT)

            if roi_start_row < roi_end_row and roi_start_row >= 0 and roi_end_row <= height and width > 0:
                roi_frame = frame_to_process[roi_start_row:roi_end_row, :]

                if roi_frame.shape[0] > 0 and roi_frame.shape[1] > 0:
                    input_tensor = preprocess_image(roi_frame, ONNX_INPUT_SIZE)
                    original_roi_shape = roi_frame.shape[:2]

                    if session and input_name and output_names:
                         # --- Inference ---
                         inf_start_time = time.time()
                         outputs = session.run(output_names, {input_name: input_tensor})
                         inf_end_time = time.time()
                         if is_first_inference and 'TensorrtExecutionProvider' in session.get_providers():
                             print(f"First inference (TensorRT engine build?) took: {(inf_end_time - inf_start_time)*1000:.2f} ms")
                             is_first_inference = False # Clear flag
                    else:
                         outputs = []

                    # Returns np.array or None
                    detections_roi_boxes = postprocess_output(outputs, CONFIDENCE_THRESHOLD, original_roi_shape)

                    if detections_roi_boxes is not None and detections_roi_boxes.size > 0:
                        # Adjust coordinates
                        adjusted_xyxy = detections_roi_boxes.copy()
                        adjusted_xyxy[:, 1] += roi_start_row
                        adjusted_xyxy[:, 3] += roi_start_row
                        # Clip to full frame
                        adjusted_xyxy[:, 0] = np.clip(adjusted_xyxy[:, 0], 0, width - 1)
                        adjusted_xyxy[:, 1] = np.clip(adjusted_xyxy[:, 1], 0, height - 1)
                        adjusted_xyxy[:, 2] = np.clip(adjusted_xyxy[:, 2], 0, width - 1)
                        adjusted_xyxy[:, 3] = np.clip(adjusted_xyxy[:, 3], 0, height - 1)
                        adjusted_boxes_result = adjusted_xyxy # Store the final boxes

            # --- Log detection time ---
            worker_end_time = time.time()
            detection_time_ms = (worker_end_time - worker_start_time) * 1000
            print(f"Detection cycle time: {detection_time_ms:.2f} ms") # Log latency

            # --- Put results (np.array or None) into the results queue ---
            try:
                results_queue.put_nowait(adjusted_boxes_result)
            except queue.Full:
                pass

        except Exception as e:
            print(f"Error in inference worker loop: {e}")
            traceback.print_exc()
            try:
                results_queue.put_nowait(None) # Put None on error
            except queue.Full:
                pass

    print("Inference worker thread stopped.")


# --- Annotator ---
# No longer needed
# box_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color.RED)

# --- Main Display Loop ---
def main():
    # --- Add check after ONNX init ---
    if session is None:
        print("ERROR: ONNX session failed to initialize. Exiting.")
        return

    print("--- Starting RTSP Connection ---") # DEBUG PRINT
    print(f"Connecting to RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    print("VideoCapture object created.") # DEBUG PRINT

    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream after creating VideoCapture object.")
        print("Please check the URL, network connectivity, and camera status.")
        return
    print("Stream opened successfully via VideoCapture.isOpened(). Starting processing loop...") # DEBUG PRINT

    window_name = "RTSP Stream Detection (Threaded - OpenCV Draw - FP32)" # Updated window title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()

    latest_boxes = None # Store the most recent boxes (np.array or None)
    frame_count = 0 # DEBUG PRINT

    while not stop_event.is_set():
        # print(f"Main loop iteration {frame_count}. Reading frame...") # DEBUG PRINT (can be very verbose)
        try:
            ret, frame = cap.read()
            # print(f"Frame read attempt {frame_count}: Success={ret}, Frame is None={frame is None}") # DEBUG PRINT

            if not ret or frame is None:
                print(f"Error: Failed to grab frame {frame_count} from stream. Stopping...")
                stop_event.set()
                break

            # --- Put latest frame into queue ---
            frame_copy = frame.copy()
            try:
                 frame_queue.put_nowait(frame_copy)
            except queue.Full:
                 pass

            # --- Check for latest results ---
            try:
                latest_boxes = results_queue.get_nowait() # Get np.array or None
            except queue.Empty:
                pass # No new results, keep using previous 'latest_boxes'

            # --- Annotate the *current live frame* with the *latest available* boxes ---
            annotated_frame = frame # Start with the live frame
            if latest_boxes is not None and latest_boxes.size > 0:
                try:
                    # Iterate through boxes and draw using cv2.rectangle
                    for box in latest_boxes:
                        x1, y1, x2, y2 = box.astype(int) # Convert to int for drawing
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                except Exception as e:
                    print(f"ERROR: Unexpected error during OpenCV drawing: {e}")
                    traceback.print_exc() # Print traceback for drawing errors

            # --- Display frame ---
            cv2.imshow(window_name, annotated_frame)

            # --- Check for exit key ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exit key 'q' pressed. Stopping...")
                stop_event.set()
                break

            frame_count += 1

        except KeyboardInterrupt:
            print("Interrupted by user (Ctrl+C). Stopping...")
            stop_event.set()
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            traceback.print_exc()
            stop_event.set()
            break

    # --- Cleanup ---
    print("Waiting for inference thread to finish...")
    inference_thread.join(timeout=1.0)
    print("Releasing video capture and destroying windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("Cleanup finished.")

if __name__ == "__main__":
    main()
