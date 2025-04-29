# Jetson orin nano setup
1. before flashing the image, check Jetson [UEFI Firmware version](https://www.jetson-ai-lab.com/initial_setup_jon.html#__tabbed_1_1)
2. follow the instructions to upgrade version or flash jetpack
3. run ```sudo pip install -U jetson-stats``` to install jtop and reboot jetson
4. run ```jtop``` to check the jetpack version
5. run ```sudo apt update``` and ```sudo apt upgrade``` and reboot jetson 

# Cuda, CuDNN
- run ```sudo apt install invidia-jetpack```
- run ```sudo jetson_release``` to check all cuda toolkits
- run ```nvcc -V``` to check if get cuda installed correctly
- if not, run follows:
  1. ```sudo vim ~/.bashrc```
  2. ```export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64:$LD_LIBRARY_PATH```
  3. ```export PATH=/usr/local/cuda-X.X/bin:$PATH```
  4. ```export CUDA_HOME=$CUDA_HOME:/usr/local/cuda```
  5. ```source ~/.bashrc```
- check cudnn, run ```ls /usr/local/cuda/include/cudnn.h``` and ```cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2```

# Torch to ONNX
 - check torch2onnx.ipynb

# ONNX to TensorRT
Check https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html

1. Download the TensorRT (Skip to 3)
2. Pip
- run ```python3 -m pip install --upgrade pip```
- run ```python3 -m pip install wheel```
- run ```python3 -m pip install --upgrade tensorrt```
- run ```python3 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12```

Watch out your cu version 12/11.

3. Build trtexec
Check https://forums.developer.nvidia.com/t/bash-trtexec-command-not-found/127302

- run ```cd to TensorRT directory```, e.g. /usr/src/tensorrt/samples/trtexec/
- run ```make```
- the trtexec should be in /usr/src/tensorrt/bin
- run ```alias trtexec="/usr/src/tensorrt/bin/trtexec"```
- run ```trtexec --onnx=model.onnx --saveEngine=test_fp16.trt --fp16``` OR
- run ```trtexec --onnx=model.onnx --shapes=pixel_values:1x3x640x640 --saveEngine=test_fp16_int8_onnx.trt --fp16 --int8``` replace pixel_values with your input name, and your image size
