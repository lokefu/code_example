# Background
Fine-tune DETR for Object detection (bounding box)


# Set-up
## Environment
python 3.13.2

## Libraries
    pip install -r requirements.txt
if in jupyter:

    pip install ipykernel

follow `all.ipynb`

## Files
Put everything in the same path.

    .
    ├── fkckpt                  # fine-tune checkpoints folder
    ├── detr                    # facebook detr's repo (run all.ipynb)
    ├── gt_cust                 # ground truth visualization
    ├── dataset.ipynb           # how to format the data (skip if directly downloading)
    ├── data_merge              # coco data for side view
    ├── pred_vis                # the visualization of inference
    ├── data_neg                # negative data
    ├── comparison              # compared visualization of inference with grount truth
    ├── finetune.ipynb          # finetune detr
    ├── all.ipynb               # preparation + finetune + eval
    ├── eval.ipynb              # inference on side view
    ├── requirements.txt
    ├── util.py
    └── README.md
