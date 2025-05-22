# Background
Object tracking with or without pre-processed detections


# PuTR

## Installation

1. Install the repo: https://github.com/chongweiliu/PuTR/tree/main

Same as:

```shell
conda create -n PuTR python=3.10  # create a virtual env
conda activate PuTR               # activate the env
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# The PyTorch version must be greater than 2.0.
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install opencv-python lap
```

2. pip install rfdetr

3. run script

## Files
- pure.py                     # do tracking with rfdetr as detection model
- pure_with_det.py            # do tracking with your own det (skip zero/non detection frames)
- pure_batch.py               # do batch tracking with your own det (skip)

## Inputs Format to PuTR
- det_bboxes = np.array([x1, y1, x2, y2]).astype(np.float32)
- det_scores = np.array(float_score).astype(np.float32)
- det_labels = np.ones(len(detections['scores']), dtype=int) or array of true integer labels


# SAMv2

## Installation

1. git clone https://github.com/heyoeyo/muggled_sam.git
2. cd muggled_sam
3. put everything from samv2 folder into muggle_sam; git apply patch.txt
4. pip install -r requirements.txt
5. `pip install cupy-cuda12x cucim-cu12` for cuda 12
6. `pip install cupy-cuda11x cucim-cu11` for cuda 11
7. Download model sam2.1_hiera_base_plus https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
8. Put model into model_weights folder
9. Create inputs folder, and put video you want inside
10. Check ./simple_examples/mot.py or ./simple_examples/mot_hun.py
    - Former uses mask IoU for tracking comparison, latter uses Hungarian matching
    - Hungarian is faster than mask IoU by 10 to 15% without sacrificing too much performance
11. Change path and settings and run
12. pip install rfdetr
13. run script

## Files
- mot.py                      # do tracking with rfdetr
- mot_hun.py
- sam_own_det.ipynb           # do tracking with own detection, same input format as putr
- skip.ipynb                  # skip non-/zero- detection frames
- skip_omit.ipynb             # run segmentation on selected frames and skip non-/zero- detection frames

## References
SAMv2: https://github.com/heyoeyo/muggled_sam


# Compare

|     | PuTR|	SAM|
|-----|-----|------|
|Video Length| > 12 hrs| 1 hr|
|FPS (whole process)| 16.6|	0.87|
|Pros| Fast (track down only 30 frames); able to handle long video;| Higher Precision (double pick-up from segmentation and detection);|
|Cons| lingering bounding boxes (the boxes kept for a while after people disappearing, almost layback for each people but most just few frames) - MISS_TOLERANCE parameter adjusting;| Slow (memory-encoding); less than 1hr video (need re-id if longer); Plenty of wrong detections due the SAM (recognize  models/top-cloth as human);|