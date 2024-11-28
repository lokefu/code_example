from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import json
from langchain_text_splitters import RecursiveJsonSplitter
from pathlib import Path
from langchain.retrievers import EnsembleRetriever
import gradio as gr
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import pickle
import re
from langchain_core.documents import Document

######## RAG ########
file_path='data.json'
data = json.loads(Path(file_path).read_text())
splitter = RecursiveJsonSplitter(max_chunk_size=300)
documents = splitter.create_documents(texts=[data])

# retriever
db1 = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
retriever1 = db1.as_retriever(search_type='mmr', search_kwargs={"k": 2})
    
db2 = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever2 = db2.as_retriever(search_type='mmr', search_kwargs={"k": 2})

db3 = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3",
                                       model_kwargs={'trust_remote_code': True}))
retriever3 = db3.as_retriever(search_type='mmr', search_kwargs={"k": 2})
    
# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[retriever1, retriever2, retriever3])

# animal list doc
with open('animal_list.pkl', 'rb') as f:
    animal_list = pickle.load(f)
animal_sum = {'the list of animals:': str(animal_list)}
#splitter = RecursiveJsonSplitter()#max_chunk_size=300)
#documents = splitter.create_documents(texts=[animal_sum])

#testdb = FAISS.from_documents(documents, HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3",
                                       #model_kwargs={'trust_remote_code': True}))
#testretriever = testdb.as_retriever(search_type='mmr', search_kwargs={"k": 1})
#animal_list_doc = testretriever.invoke('') #a list
document = Document(page_content=f'{animal_sum}',metadata={})
animal_list_doc = [document]

# llm model
mistral_models_path = ('mistral')
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

def llm(input):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=input)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=10000, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result

def decode_unicode_escape(text):
    return text.encode().decode('unicode_escape')

def decode(text):
    #extract_unicode_escape
    escaped_sequences = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '\\u':
            end_index = text.find('"', i + 2)
            if end_index != -1:
                escaped_sequences.append(text[i:end_index])
                i = end_index
            else:
                break
        else:
            i += 1
    for i in escaped_sequences:
        text = text.replace(i, decode_unicode_escape(i))
    return text #escaped_sequences, text

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Answer the following question based only on the provided context.
Some of the context might be relevant to the question, but some might not be.
Please respond to questions in a conversational and easy-to-understand manner.

<context>
{context}
</context>

Answer the following question:

{question}"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag(message):
    doc = ensemble_retriever.invoke(message)
    doc_copy = doc.copy()
    doc_copy.append(animal_list_doc[0])
    context = format_docs(doc_copy)
    prompt = RAG_TEMPLATE.format(context=context, question=message)
    answer = llm(prompt)
    
    return answer, doc

######## functions ########
def get_all_from_context(context):
    temp_dict = json.loads(context[0].model_dump()['page_content'])
    doc = next(iter(next(iter(temp_dict.values())).values()))[0]
    
    #image
    image_url = doc['image_links']
    image_path = []
    for i in image_url:
      img_name = '-'.join(i.split('/')[-2:])
      path = 'images/' + img_name
      image_path.append(path)
    unique_image_path = list(set(image_path))
    
    #name
    name = doc['Name']
    sci_name = doc['Scientific Name']
    name_output = f"Name: {name}; Scientific Name: {sci_name}"
    
    #web
    web = doc['link']
    
    return unique_image_path, name_output, web

def find_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r'https://biome\.nparks\.gov\.sg/Help/SpeciesDetail.*?\d+'
    # Find all URLs in the text
    urls = re.findall(url_pattern, text)
    tmp = []
    if urls:
        #print("Web link found in the text:")
        for url in urls:
            if url.endswith('.'):
                url = url[:-1]
            tmp.append(url)
        
        unique_list = list(set(tmp))
        return unique_list
    else:
        return None

def find_img_urls(text):
    # Regular expression pattern to match URLs
    img_pattern = r'https://biome\.nparks\.gov\.sg/Content.*?\.jpg'
    # Find all URLs in the text
    urls = re.findall(img_pattern, text)
    tmp = []
    if urls:
        #print("Web link found in the text:")
        for url in urls:
            if url.endswith('.'):
                url = url[:-1]
            tmp.append(url)
        
        unique_list = list(set(tmp))
        return unique_list[0] #return the first img, string
    else:
        return None

def get_image_from_urls(urls, context):
    image_path = []
    for i in urls:
        for j in context:
            temp_dict = json.loads(j.model_dump()['page_content'])
            each_animal = next(iter(next(iter(temp_dict.values())).values()))[0]
            if each_animal['link'] == i:
                image_url = each_animal['image_links']
                for k in image_url:
                    img_name = '-'.join(k.split('/')[-2:])
                    path = 'images/' + img_name
                    image_path.append(path)
    unique_image_path = list(set(image_path))
    
    return unique_image_path

def get_image_from_img_urls(img_url, context):
    image_path = []
    for j in context:
        temp_dict = json.loads(j.model_dump()['page_content'])
        each_animal = next(iter(next(iter(temp_dict.values())).values()))[0]
        image_list = each_animal['image_links']
        for k in image_list:
            if k == img_url:
                for k in image_list:
                    img_name = '-'.join(k.split('/')[-2:])
                    path = 'images/' + img_name
                    image_path.append(path)
                break
    unique_image_path = list(set(image_path))
    
    return unique_image_path

def check_name_from_context_in_answer(context, bot_message):
    #input - context: list of documents
    #output - the image path list
    for i in context:
        temp_dict = json.loads(i.model_dump()['page_content'])
        doc = next(iter(next(iter(temp_dict.values())).values()))[0]
        name = doc['Name']
        if name in bot_message:
            image_url = doc['image_links']
            image_path = []
            for i in image_url:
                img_name = '-'.join(i.split('/')[-2:])
                path = 'images/' + img_name
                image_path.append(path)
            unique_image_path = list(set(image_path))
            
            return unique_image_path
    return None

######## demo ########
# Load list and dictionary from file
with open('dropdown.pkl', 'rb') as f:
    species_list = pickle.load(f)
    sub_dict = pickle.load(f)
    animal_dict = pickle.load(f)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Welcome!
        Ask any questions below to know about animals.
    
        You can ask about the animal's name, habit, etc.
        You could directly enter questions or select the animal from the dropdown list.
        """)
    
    
    ### Chatbot
    chatbot = gr.Chatbot(type="messages")
    def respond(message, chat_history):
        llm_response = rag(message)
        answer, context = llm_response[0], llm_response[1]
        bot_message = decode(answer) #decode_unicode_escape(answer)
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        urls = find_urls(bot_message)
        if urls == None: #rule 1, web link in answer
            img_url = find_img_urls(bot_message)
            if img_url == None: #rule 2, image link in answer
                name_in_answer = check_name_from_context_in_answer(context, bot_message)
                if name_in_answer == None: #rule 3, animal name in retrival in answer
                    #rule 4, if all no, use top one
                    # extract the documents from the pool
                    retriever4 = BM25Retriever.from_documents(context)
                    retriever4.k = 1
                    doc_selected = retriever4.invoke(answer) #document
                    #toprint
                    image_path, _, _ = get_all_from_context(doc_selected)
                    print('rule 4: image from doc')
                else:
                    image_path = name_in_answer
                    print('rule 3: find name in answer')
            else:
                image_path = get_image_from_img_urls(img_url, context)
                print('rule 2: find img_url in answer')
        else:
            url = [urls[0]] #use only first web_ink
            image_path = get_image_from_urls(url, context)
            print('rule 1: find url in answer')

        for i in image_path:
            chat_history.append({"role": "assistant", "content": gr.Image(i, show_download_button=True)})
        #time.sleep(2)
        return "", chat_history#, name_output, web
    
    gr.Markdown(
        """
        ## Your question here
        Type a question or select the species -> sub-species -> animal from the Animal Selection below.
        Some examples questions are also provided in the Example Questions below.
        
        Remember to click the Submit button or press 'Enter' to get the answer, even if you have selected below.
        """)
    msg = gr.Textbox(label = 'Question here', placeholder = "tell me about Manthey's Chorus Frog")
    submit_btn = gr.Button("Submit")
    
    
    ### Select the animal
    with gr.Accordion("Animal Selection", open=False):
        gr.Markdown(
            """   
            Your selected choice will be only updated in [Question here] box above after you choose the animal, so please select question if need at first.
            You can still edit the question manually after selecting the animal.
            """)
        questions = gr.Dropdown(["", "tell me about ", "what is the habit of "], value='', label="Questions", info="You could skip if you don't want")
        #species
        species = gr.Dropdown(species_list, value=None, label="Species", info="Sub-species will be displayed based on the selected species")
        #sub_species
        sub = gr.Dropdown(value=None, label="Sub-species", visible=True, info="Animals will be displayed based on the selected species")
        #animal
        animal = gr.Dropdown(value=None, label="Animals", visible=True, info="You can ask questions about animals")

    ### questions examples
    with gr.Accordion("Example Questions", open=False):
        gr.Markdown(
            """
            Your selected example question will be updated in the [Question here] box above.
            Specific questions towards the animal will be answered fast, while answering generic questions takes more time.
            """
            )
        examples = gr.Examples(examples = [["tell me about Crab-eating Frog"],
                                ["what is the habit of Obelisk Creeper Snail"],
                                ["chinese name of little grebe"], 
                                ["tamil name of Javan Myna"],
                                ["website link of Grey Sailor"],
                                ["image link of Chinese Egret"],
                                ["tell me some species of birds in singapore"]], label="Examples", inputs = msg)
    
    def select_sub(species):
        return gr.Dropdown(sub_dict[species], interactive=True)
    def select_animal(sub):
        return gr.Dropdown(animal_dict[sub], interactive=True)
    def species_clear_animal(species):
        return gr.Dropdown(choices=[], value=None, interactive=True)
    
    # Function to link dropdown_animal-question to the chatbot
    def dropdown_to_bot(questions, animal):
        return f"{questions}{animal}"
    
    species.input(select_sub, species, sub)
    species.input(species_clear_animal, species, animal)
    sub.input(select_animal, sub, animal)
    animal.input(dropdown_to_bot, [questions, animal], [msg])
    
    
    ### final
    #gr.Markdown('''## Keys about the animal''')
    #img = gr.Gallery(label = 'Image(s) about the answer')
    #name = gr.Textbox(label = "Animal's name & scientific name of the image(s)")
    #web = gr.Textbox(label = 'The website of the animal in the image(s)')
    clear = gr.ClearButton([msg, chatbot, questions, species, sub, animal])#, name, web])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])#, name, web])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])#, name, web])
    
#demo.launch(share=True)
demo.launch(share=True, share_server_address="ai-demo.tictag.io:7000")
