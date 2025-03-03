# Aims
Built a multi-camera tracking system, convert bounding boxes with global id into a united 2D-plane, showcasing how objects moving

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

sample input: 

    - for data, json for bounding box, dimension: frame no -> global id no -> bounding box

sample output: 

## Outputs
The output 'GIF' or 'mp4' in 'output' folder.
