# Background
1. Singapore National Park Biodiversity, Chatbot for animals' information
2. A demo for RAG + LLM
3. Framework: Langchain
4. Text encoder/Embedding model:

        # semantic:
        sentence-transformers/all-mpnet-base-v2
        sentence-transformers/all-MiniLM-L6-v2

        # latin:
        jinaai/jina-embeddings-v3
6. Vector Database: FAISS
7. LLM: mistralai/Mistral-7B-Instruct-v0.3


# Set-up
## Environment
Python 3.10.0

## Libraries
    pip install -r requirements.txt
if in jupyter:

    pip install ipykernel

## Datas
Put everything in the same path.

    .
    ├── images                  # image folder
    ├── mistral                 # mistral LLM model folder
    ├── data.json               # reformatted data
    ├── animal_data.json        # data from web  
    ├── dropdown.pkl            # data for dropdown
    ├── demo.py
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


# Remarks
1. Will add the dropdown into demo later.

# Extra stuffs
        def get_image_from_context(context):
            temp_dict = json.loads(context[0].model_dump()['page_content'])
            image_url = next(iter(next(iter(temp_dict.values())).values()))[0]['image_links']
            image_path = []
            for i in image_url:
              img_name = '-'.join(i.split('/')[-2:])
              path = 'images/' + img_name
              image_path.append(path)
            return image_path

        def get_name_from_context(context):
            temp_dict = json.loads(context[0].model_dump()['page_content'])
            name = next(iter(next(iter(temp_dict.values())).values()))[0]['Name']
            sci_name = next(iter(next(iter(temp_dict.values())).values()))[0]['Scientific Name']
            return f"Name: {name}; Scientific Name: {sci_name}"

        def get_web_from_context(context):
            temp_dict = json.loads(context[0].model_dump()['page_content'])
            web = next(iter(next(iter(temp_dict.values())).values()))[0]['link']
            return web
