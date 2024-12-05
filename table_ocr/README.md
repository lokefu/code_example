# Background
1. Table recognition and OCR from PDF
2. Table location (page) by keyword match via RAG database
3. Table recognition (in each page) and OCR by pymupdf library
3. Framework: Langchain


# Set-up
## Environment
Python 3.9.20

## Libraries
    pip install -r requirements.txt
if in jupyter:

    pip install ipykernel

## Files
Put everything in the same path.

    .
    ├── 1/2/3/4.png             # example images
    ├── data/example.pdf        # example PDFs
    ├── example.pdf-X-X.png     # (output) detected images in PDF
    ├── x.pdf                   # (output) bounding box drawn in PDF  
    ├── pymupdf.ipynb           # code exploration
    ├── name.py                 # example to get animal names in table in PDF (with certain rule-based)
    ├── name.pkl                # (output) name list
    ├── requirements.txt
    └── README.md

## Run
    python3 demo.py

## Outputs
`name.pkl` will be outputed if running `name.py`.

To load result:

    import pickle

    # Load list and dictionary from file
    with open('name.pkl', 'rb') as f:
        names = pickle.load(f)
