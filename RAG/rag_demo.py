import os
import json
import pickle
import re
from pathlib import Path
from typing import List, TypedDict

import gradio as gr
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# LangGraph 核心
from langgraph.graph import END, StateGraph

# Mistral 原生推理库
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

######## 1. 基础配置与模型加载 ########

# 数据加载与切分
file_path = 'data.json'
data = json.loads(Path(file_path).read_text())
splitter = RecursiveJsonSplitter(max_chunk_size=300)
documents = splitter.create_documents(texts=[data])

# 初始化 Mistral 模型 (用于推理和查询改写)
mistral_models_path = 'mistral'
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

def llm_inference(prompt_text):
    """封装原有的 Mistral 推理逻辑"""
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt_text)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=2048, temperature=0.0, 
                             eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

# 定义 LangChain 兼容的 LLM 接口 (用于 MultiQueryRetriever)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, AIChatMessage

class MistralLLM(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        res = llm_inference(messages[0].content)
        return ChatResult(generations=[ChatGeneration(message=AIChatMessage(content=res))])
    @property
    def _llm_type(self): return "mistral_local"

custom_llm = MistralLLM()

######## 2. 构建 Advanced Retrieval 链路 ########

# 多 Embedding 集成检索 (维持原有逻辑)
embeddings1 = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
db1 = FAISS.from_documents(documents, embeddings1)
retriever1 = db1.as_retriever(search_type='mmr', search_kwargs={"k": 3})

embeddings3 = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3", model_kwargs={'trust_remote_code': True})
db3 = FAISS.from_documents(documents, embeddings3)
retriever3 = db3.as_retriever(search_type='mmr', search_kwargs={"k": 3})

# 集成检索器
ensemble_base = EnsembleRetriever(retrievers=[retriever1, retriever3], weights=[0.5, 0.5])

# 引入 Multi-Query: 将用户问题改写为 3 个不同版本，扩大搜索面
mq_retriever = MultiQueryRetriever.from_llm(retriever=ensemble_base, llm=custom_llm)

# 引入 Flashrank Re-ranking: 对初步检索到的文档进行精排，保留最相关的 Top-3
reranker = FlashrankRerank()
advanced_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=mq_retriever)

# 准备静态文档 (animal_list)
with open('animal_list.pkl', 'rb') as f:
    animal_list = pickle.load(f)
animal_list_doc = Document(page_content=f'the list of animals: {str(animal_list)}', metadata={})

######## 3. LangGraph 工作流设计 ########

class GraphState(TypedDict):
    question: str
    context: List[Document]
    generation: str

def retrieve_node(state: GraphState):
    """节点1: 执行 Multi-Query + Ensemble + Rerank 检索"""
    print(f"--- 正在检索并重排序: {state['question']} ---")
    docs = advanced_retriever.invoke(state['question'])
    # 结合静态 animal_list
    all_docs = docs + [animal_list_doc]
    return {"context": all_docs}

def generate_node(state: GraphState):
    """节点2: 生成回答"""
    print("--- 正在生成回答 ---")
    context_text = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = f"""You are an assistant for question-answering tasks. Answer based ONLY on context.
    Context: {context_text}
    Question: {state['question']}
    Answer:"""
    answer = llm_inference(prompt)
    return {"generation": answer}

# 编排工作流
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
rag_app = workflow.compile()

######## 4. 工具函数 (维持原有逻辑) ########

def decode(text):
    # 处理 Unicode 转义字符
    try:
        return text.encode().decode('unicode_escape')
    except:
        return text

def find_urls(text):
    url_pattern = r'https://biome\.nparks\.gov\.sg/Help/SpeciesDetail.*?\d+'
    return list(set(re.findall(url_pattern, text)))

def find_img_urls(text):
    img_pattern = r'https://biome\.nparks\.gov\.sg/Content.*?\.jpg'
    urls = re.findall(img_pattern, text)
    return urls[0] if urls else None

def get_image_path_from_doc(doc):
    try:
        temp_dict = json.loads(doc.page_content)
        # 兼容你的 JSON 嵌套结构
        inner_data = next(iter(next(iter(temp_dict.values())).values()))[0]
        image_urls = inner_data.get('image_links', [])
        return [f"images/{'-'.join(url.split('/')[-2:])}" for url in image_urls]
    except:
        return []

######## 5. Gradio UI 响应逻辑 ########

def respond(message, chat_history):
    # 执行 LangGraph RAG
    result = rag_app.invoke({"question": message})
    answer = decode(result["generation"])
    context_docs = result["context"]
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})
    
    # 图像匹配逻辑 (优化版)
    image_paths = []
    # 策略：如果回答中提到了 URL 或图片链接，优先从 Context 查找对应图片
    urls = find_urls(answer)
    img_url_in_text = find_img_urls(answer)
    
    if urls or img_url_in_text:
        # 遍历文档寻找匹配的 URL
        for doc in context_docs:
            if any(u in doc.page_content for u in (urls + ([img_url_in_text] if img_url_in_text else []))):
                image_paths.extend(get_image_path_from_doc(doc))
    
    # 如果没找到，退而求其次使用检索到的第一个文档的图片
    if not image_paths and context_docs:
        image_paths = get_image_path_from_doc(context_docs[0])

    # 在 Chatbot 中显示图片
    for path in list(set(image_paths)):
        if os.path.exists(path):
            chat_history.append({"role": "assistant", "content": gr.Image(path)})
            
    return "", chat_history

######## 6. Gradio UI 界面 (维持原有布局) ########

with gr.Blocks() as demo:
    gr.Markdown("# 🐾 Singapore Biodiversity AI Assistant")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox(label="Ask about animals", placeholder="Tell me about Crab-eating Frog...")
    submit_btn = gr.Button("Submit")
    
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    # ... 原有的 Dropdown 和 Examples 逻辑可以按需添加 ...

if __name__ == "__main__":
    demo.launch()