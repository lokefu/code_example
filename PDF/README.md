# Aims
To extract information from PDF's table, then use RAG for generation

# Background
1. Data extraction from PDF's table
2. RAG with keywords format match + LLM for summary and generation
3. Framework: Langchain
4. Text encoder/Embedding model:

        # semantic:
        sentence-transformers/all-mpnet-base-v2

6. Vector Database: FAISS
7. LLM: mistralai/Mistral-7B-Instruct-v0.3


# Set-up
## Environment
Python 3.10.0

## Libraries
    pip install -r requirements.txt
if in jupyter:

    pip install ipykernel

## Files
Put everything in the same path.

    .
    ├── mistral                 # mistral LLM model folder 
    ├── result.pkl              # LLM sample response
    ├── data.pdf                # PDF
    ├── demo.py                 # a demo built for data.pdf
    ├── test.ipynb
    ├── requirements.txt
    └── README.md

1. All except mistral folder are in github.
2. To download the mistral: 1) https://drive.google.com/drive/folders/1OaPUWUSG4OPRYOBzNY01Tx57C8CIkFZ9?usp=sharing; 2) use code below. (If you want to change the folder path of model, remember to change it too in the demo.py - mistral_models_path = ('path_to_model'))

        from huggingface_hub import snapshot_download
        from pathlib import Path
        from huggingface_hub import login
        login('hf_token') #set the token with WRITE access right, not READ only

        mistral_models_path = ('mistral')

        snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
