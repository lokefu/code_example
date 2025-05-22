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
 - check ```torch2onnx.ipynb``` - torch to onnx by torch

# ONNX to TensorRT
Check https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html

1. Download the TensorRT (Prefered, Skip to 3)
2. pip
- run ```python3 -m pip install --upgrade pip```
- run ```python3 -m pip install wheel```
- run ```python3 -m pip install --upgrade tensorrt```
- run ```python3 -m pip install tensorrt-cu12 tensorrt-lean-cu12 tensorrt-dispatch-cu12```

Watch out your cu version 12/11.

3. Generate

  - 3.1. Build trtexec
  Check https://forums.developer.nvidia.com/t/bash-trtexec-command-not-found/127302
  
    - run ```cd to TensorRT directory```, e.g. /usr/src/tensorrt/samples/trtexec/
    - run ```make```
    - the trtexec should be in /usr/src/tensorrt/bin
    - run ```alias trtexec="/usr/src/tensorrt/bin/trtexec"```
    - run ```trtexec --onnx=model.onnx --saveEngine=test_fp16.trt --fp16``` OR
    - for some models cannot be recognized by trt or for dynamic input purpose, need to input --shapes, run ```trtexec --onnx=model.onnx --shapes=pixel_values:1x3x640x640 --saveEngine=test_fp16_int8_onnx.trt --fp16 --int8``` replace pixel_values with your input name, and your image size

  - 3.2. Use ONNX cache (Cheng Wee's)
  
    - see ```onnx2trt.py```
    - Install ```onnxruntime-gpu``` by ```pip install -r requirements.txt --index-url your_url```
    - index-url at https://pypi.jetson-ai-lab.dev/ (select your jp and cu versions)


# Benchmark (TensorRT)
- datset based on human detection (crowd.zip)
- model: PekingU/rtdetr_r101vd_coco_o365

|Model| threshold|	mAP50| Precision|
|-----|----------|-------|----------|
|Pre-trained R50|	0.6|	0.22|	0.92|
|Pre-trained R50|	0.3|	0.35|	0.77|

- FPS

|Model|	Avg. Latency (s/image)| Model Size (torch/onnx)| Image Size|
|-----|-----------------------| -----------------------| ----------|
|PT| 0.15| 300MB| 640|
|FT| 0.25| 300MB| 960|

# Remark
Upgrade jetson jetpack: https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/SoftwarePackagesAndTheUpdateMechanism.html#updating-a-jetson-device

Nvidia jetson python library version: https://pypi.jetson-ai-lab.dev/
