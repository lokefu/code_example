# Background
1. Multi-Camera Videos to 2D mapping
2. Homography estimation
3. Object Detection


# Set-up
## Environment
Python 3.9.2

## Libraries
    pip install -r requirements.txt
if in jupyter:

    pip install ipykernel

## Files
Put everything in the same path.

    .
    ├── input                   # input folder
        ├── video                   # video folder
        ├── data                    # track data folder
        ├── map                     # map folder
    ├── map_multi.py            # functions for multi-camera
    ├── map_single.py           # functions for single-camera
    ├── util.py                 # util functions
    ├── multi.ipynb             # notebook for multi-camera
    ├── exploration.ipynb       # code exploration
    ├── single.ipynb            # notebook for single-camera
    ├── requirements.txt
    ├── output                  # output folder
    └── README.md

# Result
## Run
    multi.ipynb

    single.ipynb

sample input: https://drive.google.com/drive/folders/1gW9BfKYLnrPUUcWsOsCqBqh440DMrnRl?usp=drive_link

sample output: https://drive.google.com/drive/folders/1LxTQKhU2Mrwp6lzuyLbg9dkpaHy-msH1?usp=drive_link

## Outputs
The output 'GIF' or 'mp4' in 'output' folder.
