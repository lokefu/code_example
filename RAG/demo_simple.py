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

mistral_models_path = ('mistral')
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

def llm(input):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=input)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=10000, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result

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
    context = format_docs(doc)
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
    
    #name
    name = doc['Name']
    sci_name = doc['Scientific Name']
    name_output = f"Name: {name}; Scientific Name: {sci_name}"
    
    #web
    web = doc['link']
    
    return image_path, name_output, web


######## demo ########
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Welcome!
    Ask any questions below to know about animals.
    """)
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label = 'Ask a question here', placeholder = "tell me about Manthey's Chorus Frog")
    #img = gr.Gallery(label = 'Image(s) about the answer')
    doc = gr.Textbox(label = "Animal's name & scientific name of the image(s)")
    web = gr.Textbox(label = 'The website of the animal in the image(s)')
    clear = gr.ClearButton([msg, chatbot, doc])
    submit_btn = gr.Button("Submit")
    

    def respond(message, chat_history):
        llm_response = rag(message)
        answer, context = llm_response[0], llm_response[1]
        bot_message = answer
        
        # extract the documents from the pool
        retriever4 = BM25Retriever.from_documents(context)
        retriever4.k = 1
        doc_selected = retriever4.invoke(answer) #document
        
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        
        #toprint
        image_path, name_output, web = get_all_from_context(doc_selected)
        for i in image_path:
            chat_history.append({"role": "assistant", "content": gr.Image(i, show_download_button=True)})
        #time.sleep(2)
        return "", chat_history, name_output, web

    msg.submit(respond, [msg, chatbot], [msg, chatbot, doc, web])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot, doc, web])
  
demo.launch(share=True, share_server_address="ai-demo.tictag.io:7000")
