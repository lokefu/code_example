import os
import json
import pickle
import re
from pathlib import Path
from typing import List, TypedDict, Annotated

import gradio as gr
from pydantic import BaseModel, Field

# LangChain Core & Retrievers
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_milvus import Milvus

# LangGraph Core & Memory
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Mistral Inference
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

######## 1. Data & Retrievers Setup (Upgraded to Milvus) ########
file_path = 'data.json'
data = json.loads(Path(file_path).read_text()) if os.path.exists(file_path) else {"sample": "data"}
splitter = RecursiveJsonSplitter(max_chunk_size=300)
documents = splitter.create_documents(texts=[data])

# Setup local Milvus database URI
milvus_uri = "./milvus_agent.db"

# Multi-Embedding Ensemble with Milvus
emb1 = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
db1 = Milvus.from_documents(documents, emb1, connection_args={"uri": milvus_uri}, collection_name="mpnet_coll", drop_old=True)
retriever1 = db1.as_retriever(search_type='mmr', search_kwargs={"k": 2})

emb2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db2 = Milvus.from_documents(documents, emb2, connection_args={"uri": milvus_uri}, collection_name="minilm_coll", drop_old=True)
retriever2 = db2.as_retriever(search_type='mmr', search_kwargs={"k": 2})

emb3 = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3", model_kwargs={'trust_remote_code': True})
db3 = Milvus.from_documents(documents, emb3, connection_args={"uri": milvus_uri}, collection_name="jina_coll", drop_old=True)
retriever3 = db3.as_retriever(search_type='mmr', search_kwargs={"k": 2})
    
ensemble_retriever = EnsembleRetriever(retrievers=[retriever1, retriever2, retriever3])

# Animal list static doc
animal_list_doc = [Document(page_content="the list of animals: []", metadata={})]
if os.path.exists('animal_list.pkl'):
    with open('animal_list.pkl', 'rb') as f:
        animal_list = pickle.load(f)
    animal_sum = {'the list of animals:': str(animal_list)}
    animal_list_doc = [Document(page_content=f'{animal_sum}', metadata={})]

######## 2. LLM Model Setup ########
mistral_models_path = 'mistral'
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

def llm_inference(input_text):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=input_text)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=2048, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

######## 3. Structured Output Definition ########
class AgentResponse(BaseModel):
    answer: str = Field(description="The conversational answer to the user's question.")
    detected_species: List[str] = Field(description="List of animal names explicitly mentioned in the context.")

json_parser = JsonOutputParser(pydantic_object=AgentResponse)

######## 4. Utilities ########
def decode_unicode_escape(text):
    return text.encode().decode('unicode_escape')

def decode(text):
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
    return text

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

######## 5. Image/URL Processing Functions ########
def get_all_from_context(context):
    try:
        temp_dict = json.loads(context[0].model_dump()['page_content'])
        doc = next(iter(next(iter(temp_dict.values())).values()))[0]
        
        image_url = doc.get('image_links', [])
        image_path = [f"images/{'-'.join(i.split('/')[-2:])}" for i in image_url]
        unique_image_path = list(set(image_path))
        
        name = doc.get('Name', '')
        sci_name = doc.get('Scientific Name', '')
        name_output = f"Name: {name}; Scientific Name: {sci_name}"
        web = doc.get('link', '')
        
        return unique_image_path, name_output, web
    except:
        return [], "", ""

def find_urls(text):
    url_pattern = r'https://biome\.nparks\.gov\.sg/Help/SpeciesDetail.*?\d+'
    urls = re.findall(url_pattern, text)
    return list(set([url[:-1] if url.endswith('.') else url for url in urls])) if urls else None

def find_img_urls(text):
    img_pattern = r'https://biome\.nparks\.gov\.sg/Content.*?\.jpg'
    urls = re.findall(img_pattern, text)
    if urls:
        tmp = [url[:-1] if url.endswith('.') else url for url in urls]
        return list(set(tmp))[0] 
    return None

def get_image_from_urls(urls, context):
    image_path = []
    for i in urls:
        for j in context:
            try:
                temp_dict = json.loads(j.model_dump()['page_content'])
                each_animal = next(iter(next(iter(temp_dict.values())).values()))[0]
                if each_animal.get('link') == i:
                    for k in each_animal.get('image_links', []):
                        image_path.append(f"images/{'-'.join(k.split('/')[-2:])}")
            except: continue
    return list(set(image_path))

def get_image_from_img_urls(img_url, context):
    image_path = []
    for j in context:
        try:
            temp_dict = json.loads(j.model_dump()['page_content'])
            each_animal = next(iter(next(iter(temp_dict.values())).values()))[0]
            image_list = each_animal.get('image_links', [])
            if img_url in image_list:
                image_path.extend([f"images/{'-'.join(k.split('/')[-2:])}" for k in image_list])
                break
        except: continue
    return list(set(image_path))

def check_name_from_context_in_answer(context, bot_message):
    for i in context:
        try:
            temp_dict = json.loads(i.model_dump()['page_content'])
            doc = next(iter(next(iter(temp_dict.values())).values()))[0]
            if doc.get('Name') and doc['Name'] in bot_message:
                return list(set([f"images/{'-'.join(url.split('/')[-2:])}" for url in doc.get('image_links', [])]))
        except: continue
    return None

######## 6. LangGraph Agent Workflow (Upgraded with Memory) ########
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    documents: List[Document]
    parsed_json: dict
    is_relevant: str

def retrieve_node(state: AgentState):
    query = state["messages"][-1].content
    doc = ensemble_retriever.invoke(query)
    doc_copy = doc.copy()
    doc_copy.extend(animal_list_doc)
    return {"documents": doc_copy}

def grade_documents(state: AgentState):
    query = state["messages"][-1].content
    context_text = format_docs(state["documents"])
    prompt = f"Grade these documents for relevancy to the question: {query}\nDocuments: {context_text}\nRespond with only 'yes' or 'no'."
    score = llm_inference(prompt).strip().lower()
    return {"is_relevant": "yes" if "yes" in score else "no"}

def generate_node(state: AgentState):
    query = state["messages"][-1].content
    context = format_docs(state["documents"])
    history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][:-1]])
    
    prompt = f"""You are an assistant for question-answering tasks. Answer based only on the context.
    
    Chat History:
    {history}
    
    Context:
    {context}
    
    Question: {query}
    
    {json_parser.get_format_instructions()}
    
    Return ONLY valid JSON:"""
    
    raw_response = llm_inference(prompt)
    try:
        # Clean up potential markdown formatting from local LLMs before parsing
        clean_json = raw_response.strip().strip('```json').strip('```').strip()
        parsed = json_parser.parse(clean_json)
    except Exception as e:
        parsed = {"answer": raw_response, "detected_species": []}
        
    return {
        "parsed_json": parsed,
        "messages": [AIMessage(content=parsed["answer"])]
    }

def fallback_node(state: AgentState):
    parsed = {
        "answer": "I couldn't find specific information about that in my current database.",
        "detected_species": []
    }
    return {
        "parsed_json": parsed,
        "messages": [AIMessage(content=parsed["answer"])]
    }

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_docs", grade_documents)
workflow.add_node("generate", generate_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_docs")
workflow.add_conditional_edges(
    "grade_docs",
    lambda x: x["is_relevant"],
    {"yes": "generate", "no": "fallback"}
)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# Compile Agent with Memory
memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)

######## 7. Gradio UI ########

# Load dropdown resources safely
species_list, sub_dict, animal_dict = [], {}, {}
if os.path.exists('dropdown.pkl'):
    with open('dropdown.pkl', 'rb') as f:
        species_list = pickle.load(f)
        sub_dict = pickle.load(f)
        animal_dict = pickle.load(f)

with gr.Blocks() as demo:
    gr.Markdown("# Welcome! Ask any questions below to know about animals.")
    
    chatbot = gr.Chatbot(type="messages")
    
    def respond(message, chat_history):
        # Configure thread for conversational memory tracking
        config = {"configurable": {"thread_id": "session_default"}}
        
        result = agent_app.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        parsed_res = result["parsed_json"]
        bot_message = decode(parsed_res["answer"])
        context = result["documents"]
        
        print(f"Detected Species in JSON: {parsed_res.get('detected_species', [])}")
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        # Original 4-Rule Image Logic
        image_path = []
        urls = find_urls(bot_message)
        if urls is None: 
            img_url = find_img_urls(bot_message)
            if img_url is None: 
                name_in_answer = check_name_from_context_in_answer(context, bot_message)
                if name_in_answer is None: 
                    if context and len(context) > 0 and context[0].page_content != "the list of animals: []":
                        retriever4 = BM25Retriever.from_documents(context)
                        retriever4.k = 1
                        doc_selected = retriever4.invoke(bot_message) 
                        if doc_selected:
                            image_path, _, _ = get_all_from_context(doc_selected)
                            print('rule 4: image from doc')
                else:
                    image_path = name_in_answer
                    print('rule 3: find name in answer')
            else:
                image_path = get_image_from_img_urls(img_url, context)
                print('rule 2: find img_url in answer')
        else:
            url = [urls[0]] 
            image_path = get_image_from_urls(url, context)
            print('rule 1: find url in answer')

        for i in (image_path or []):
            if os.path.exists(i):
                chat_history.append({"role": "assistant", "content": gr.Image(i, show_download_button=True)})
        
        return "", chat_history
    
    msg = gr.Textbox(label='Question here', placeholder="tell me about Manthey's Chorus Frog")
    submit_btn = gr.Button("Submit")
    
    with gr.Accordion("Animal Selection", open=False):
        questions = gr.Dropdown(["", "tell me about ", "what is the habit of "], value='', label="Questions")
        species = gr.Dropdown(species_list, value=None, label="Species")
        sub = gr.Dropdown(value=None, label="Sub-species", visible=True)
        animal = gr.Dropdown(value=None, label="Animals", visible=True)

    with gr.Accordion("Example Questions", open=False):
        examples = gr.Examples(examples=[
            ["tell me about Crab-eating Frog"],
            ["what is the habit of Obelisk Creeper Snail"],
            ["website link of Grey Sailor"]
        ], label="Examples", inputs=msg)
    
    def select_sub(species): return gr.Dropdown(sub_dict.get(species, []), interactive=True)
    def select_animal(sub): return gr.Dropdown(animal_dict.get(sub, []), interactive=True)
    def species_clear_animal(species): return gr.Dropdown(choices=[], value=None, interactive=True)
    def dropdown_to_bot(questions, animal): return f"{questions}{animal}"
    
    species.input(select_sub, species, sub)
    species.input(species_clear_animal, species, animal)
    sub.input(select_animal, sub, animal)
    animal.input(dropdown_to_bot, [questions, animal], [msg])
    
    clear = gr.ClearButton([msg, chatbot, questions, species, sub, animal])

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    
if __name__ == "__main__":
    demo.launch(share=True, share_server_address="ai-demo.tictag.io:7000")