import traceback # For detailed error printing
import onnxruntime as ort # Import ONNX Runtime
import sys # For exiting
import os # Needed for checking cache path

# --- Configuration ---
# *** Path to the ORIGINAL FP32 ONNX model ***
ONNX_MODEL_PATH = 'model/inference_model.sim.onnx' # <<< CHANGED BACK TO FP32 MODEL
# *** IMPORTANT: Set a valid, writable path for the TensorRT engine cache ***
TENSORRT_CACHE_PATH = '/path/to/your/trt_cache' # e.g., '/home/jetson/my_app_trt_cache/'


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