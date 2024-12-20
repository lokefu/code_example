from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import gradio as gr
import pickle
import re
import pandas as pd
import time


####### LLM #######

# 4s
mistral_models_path = ('mistral')
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

def llm(input):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=input)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=10000, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result
  

####### PDF to Database #######
def read_pdf(file_path):
    #file_path = ('data.pdf')
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    pages = loader.load_and_split(text_splitter=text_splitter)
    
    # table
    pattern = r'Table \d+\.\d+\.'
    table_list = []
    for i in range(0, len(pages)):
        if re.search(pattern, pages[i].page_content):
            table_list.append(pages[i])

    ## species name
    species_name_pattern = re.compile(r'species name', re.IGNORECASE)
    species_list = []
    for i in range(0, len(pages)):
        if re.search(species_name_pattern, pages[i].page_content):
            pages[i].metadata['species'] = 'Yes'
            species_list.append(pages[i])
        else:
            pages[i].metadata['species'] = 'No'

    ## Conservation Status
    status_pattern = r'Conservation Status'
    status_list = []
    for i in range(0, len(table_list)):
        if re.search(status_pattern, table_list[i].page_content):
            status_list.append(table_list[i])
    
    db1 = FAISS.from_documents(table_list, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    retriever = db1.as_retriever(search_type='mmr', search_kwargs={"k": 5})
    return retriever, species_list, status_list, table_list
        

####### RAG #######

RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Answer the following question based only on the provided context.
Some of the context might be relevant to the question, but some might not be.

<context>
{context}
</context>

Answer the following question:

{question}"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

####### Output #######

def get_unique_lines(answer):
    lines = answer.splitlines()
    unique_lines = []
    seen_lines = set()
    for line in lines:
        if line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)
    return unique_lines

def get_species_name(species_list):
    context = format_docs(species_list)
    prompt = RAG_TEMPLATE.format(context=context, question='Please list all species name in the context, present each name in a new line.')
    answer = llm(prompt)
    names = get_unique_lines(answer)
    return names

def get_status(status_list):
    context = format_docs(status_list)
    prompt = RAG_TEMPLATE.format(context=context, question='Please list all conservation status in the context, present each status in a new line.')
    answer = llm(prompt)
    status = get_unique_lines(answer)
    return status

def rag(message, retriever):
    doc = retriever.invoke(message)
    context = format_docs(doc)
    prompt = RAG_TEMPLATE.format(context=context, question=message)
    answer = llm(prompt)
    
    return answer


####### Demo #######
count_list = [0] #for count in general question, to start the retriever
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Welcome!
        Ask any questions about the uploaded PDF.
        
        You could select the specific concept from the dropdown list and ask questions about the concept, or directly enter general questions.
        
        The flow: Upload PDF -> Process -> Select Concept & Ask Questions OR Ask General Questions
        """)
    gr.Markdown(
            """
            ## Upload PDF
            """)
    with gr.Accordion("Upload", open=True):       
        pdf = gr.File(label="Upload PDF")
        process_btn = gr.Button("Process")
        process_tb = gr.Textbox(label = 'Progress (30s)', placeholder = "Not yet processed")
    
    def process_wait():
        time.sleep(30)
        return gr.Textbox('Process completed')

    retriever, species_list, status_list, table_list = gr.State([]), gr.State([]), gr.State([]), gr.State([])
                  
    gr.Markdown(
        """
        ## Chat Here
        """)
    ### Chatbot
    chatbot = gr.Chatbot(type="messages")
    
    
    ### Select the question
    gr.Markdown(
        """
        ### Concept Selection
        You can select the specific concept from the dropdown list below.
        """)
    with gr.Accordion("Concept", open=False):
        gr.Markdown(
            """
            The information about the selected concept will be displayed in the below table.
            
            You can ask questions about the selected concept in the below [Question] box. E.g., for species name: "What is the species name of the first species in the table?"; for conservation status: "What is the full name of 'DD?"
            
            The answer will be only based on the table's information of selected concept. The answer will be displayed in the [Chatbot] above.
            
            Remember to click the Submit button or press 'Enter' to get the answer.
            """)
        questions = gr.Dropdown(["", "Species Name", "Conservation Status"], value='', label="Concept", info="'Species Name' takes about 2 minutes.")
        msg_c = gr.Textbox(label = 'Question', placeholder = "")
        submit_btn_c = gr.Button("Submit")
    
        llm_answer = gr.State([])
        table = gr.Dataframe(headers=['Example'], col_count=(1, 'fixed'))
        def select_question(questions, species_list, status_list):
            if questions == "Species Name":
                llm_answer = get_species_name(species_list)
                df = pd.DataFrame({"Species Name" : llm_answer})
                return gr.Dataframe(df), llm_answer
            elif questions == "Conservation Status":
                llm_answer = get_status(status_list)
                df = pd.DataFrame({"Conservation Status" : llm_answer})
                return gr.Dataframe(df), llm_answer
            else:
                llm_answer = ""
                return gr.Dataframe(headers=['Example'], col_count=(1, 'fixed')), llm_answer
        
        def respond_c(doc, message, chat_history):
            prompt = RAG_TEMPLATE.format(context=doc, question=message)
            llm_response = llm(prompt)
            
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": llm_response})
            
            #time.sleep(2)
            return "", chat_history
    
    ### General Question
    gr.Markdown(
            """
            ### General Question
            You can ask any general questions about the uploaded PDF.
            """)
    with gr.Accordion("General", open=False):
        gr.Markdown(
            """
            The general questions might focus on details with clear instructions. See the following examples for reference.
            
            The answer will be displayed in the same [Chatbot] above as the Concept Question's answer.
            
            Remember to click the Submit button or press 'Enter' to get the answer, even if you have selected below.
            """)
        def respond(message, retriever, chat_history, table_list):
            if count_list[-1] == 0:
                count_list.append(1)
                db1 = FAISS.from_documents(table_list, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
                retriever = db1.as_retriever(search_type='mmr', search_kwargs={"k": 5})
            
            llm_response = rag(message, retriever)
            
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": llm_response})
            
            #time.sleep(2)
            return "", chat_history
        
        msg_g = gr.Textbox(label = 'General Question', placeholder = "")
        submit_btn = gr.Button("Submit")
        
        ### questions examples
        with gr.Accordion("Example Questions", open=False):
            gr.Markdown(
                """
                Your selected example question will be updated in the [General Question] box above.
                """
                )
            examples = gr.Examples(examples = [["Species name list of only Birds species"],
                                    ["How many species of mammals currently extant in Singapore?"],
                                    ["What are the local status and global status of Sunda pangolin, respectively?"]],
                                    label="Examples", inputs = msg_g)
    
    
    ### actions
    pdf.upload(read_pdf, pdf, [retriever, species_list, status_list, table_list])
    process_btn.click(process_wait, [], [process_tb]) #give some time for the pdf to be processed
    questions.input(select_question, [questions, species_list, status_list], [table, llm_answer])
    msg_c.submit(respond_c, [llm_answer, msg_c, chatbot], [msg_c, chatbot])
    submit_btn_c.click(respond_c, [llm_answer, msg_c, chatbot], [msg_c, chatbot])
    msg_g.submit(respond, [msg_g, retriever, chatbot, table_list], [msg_g, chatbot])
    submit_btn.click(respond, [msg_g, retriever, chatbot, table_list], [msg_g, chatbot])
    
    ### end
    clear = gr.ClearButton([msg_g, chatbot, questions, table, msg_c])
    
#demo.launch(share=True)
demo.launch(share=True, share_server_address="ai-demo.tictag.io:7000")
