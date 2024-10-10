from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from langchain.memory import ConversationBufferMemory
from chainlit.input_widget import Slider, Select, TextInput
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from langchain.schema import BaseRetriever, Document
from langchain.retrievers import EnsembleRetriever
from typing import List
import chainlit as cl

# 加载环境变量，API和BaseUrl
load_dotenv()

import os

# 打印当前工作目录
print("当前工作目录:", os.getcwd())
print("向量存储路径:", os.path.abspath('./vector_store'))

class RerankRetriever(BaseRetriever):
    docs: List[Document]  # 存储文档列表
    bm25: BM25Okapi  # BM25检索模型
    k: int = 5  # 返回的文档数量

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 使用BM25检索文档
        bm25_scores = self.bm25.get_scores(query.split())
        
        # 结合原始顺序和BM25分数
        ranked_docs = sorted(zip(self.docs, bm25_scores), key=lambda x: x[1], reverse=True)
        
        # 返回排序后的文档
        return [doc for doc, _ in ranked_docs[:self.k]]  # 返回前k个文档

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

@cl.on_chat_start
async def start():
    # 设置聊天参数，包括温度、模型和提示模板
    settings = await cl.ChatSettings(
        [
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0.5,
                min=0,
                max=2,
                step=0.1,
            ),
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-4-1106-preview"],
                initial_index=0,
            ),
            Select(
                id="PromptTemplate",
                label="选择Prompt模板",
                values=[
                    "默认模板",
                    "专业分析模板",
                    "简明总结模板",
                    "创意思考模板"
                ],
                initial_index=0,
            ),
            TextInput(id="BaseUrl", label="OpenAI API Base", initial="https://api.openai.com/v1"),
            TextInput(id="ApiKey", label="OpenAI API Key", initial="")
        ]
    ).send()

    # 请求用户上传PDF文件
    files = await cl.AskFileMessage(
        content="请上传你要提问的PDF文件",
        accept=["application/pdf"],
        max_size_mb=3
    ).send()

    if not files:
        await cl.Message(content="未收到有效文件，请重新开始对话。").send()
        return

    file = files[0]

    msg = cl.Message(content=f'正在处理: `{file.name}`...')
    await msg.send()

    file_path = f'./{file.name}'

    # 分拆文档
    with open(file_path, 'wb') as f:
        f.write(file.content)

    docs = PyMuPDFLoader(file_path).load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)

    # 创建嵌入模型
    embeddings = OpenAIEmbeddings()  # 创建嵌入模型
    
    # 检查向量存储是否存在，如存在就加载
    try:
        # 加载向量存储
        docsearch = FAISS.load_local(
            folder_path='./vector_store',
            embeddings=embeddings,
            index_name=file.name,
            allow_dangerous_deserialization=True
        )

        # 检查索引是否加载成功
        if docsearch is not None:
            print("数据集已存在")
            msg.content = f'数据集 `{file.name}` 已存在，直接加载数据。'
            await msg.send()
        else:
            print("加载的索引为空。")
            raise ValueError("加载的索引为空。")

    except Exception as e:
        print(f'数据集`{file.name}` 不存在')
        msg.content = f'数据集`{file.name}` 不存在，开始解析PDF。'
        await msg.send()

        # 创建文档向量存储，指定本地存储路径
        docsearch = await cl.make_async(FAISS.from_documents)(
            split_docs, embeddings
        )

        # 保存到本地
        docsearch.save_local(folder_path='./vector_store', index_name='report.pdf')
    
    # 创建BM25模型
    corpus = [doc.page_content for doc in split_docs]
    bm25 = BM25Okapi(corpus)

    # 创建RerankRetriever
    rerank_retriever = RerankRetriever(docs=split_docs, bm25=bm25)

    # 创建EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[rerank_retriever, docsearch.as_retriever()],
        weights=[0.5, 0.5]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    prompt_templates = {
        "默认模板": """使用以下信息和聊天历史来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

问题: {question}

回答:""",
        "专业分析模板": """请以专业的角度，基于以下信息深入分析并回答问题。如果没有相关信息，请明确说明。

问题: {question}

专业分析:""",
        "简明总结模板": """请简明扼要地总结以下信息中与问题相关的部分，并给出简洁的回答。

问题: {question}

简明总结:""",
        "创意思考模板": """请用创新的思维方式，基于以下信息来回答问题。可以提出新的见解或独特的观点，但要确保与信息相关。

问题: {question}

创意回答:"""
    }

    PROMPT = PromptTemplate(
        template=prompt_templates[settings["PromptTemplate"]],
        input_variables=["chat_history", "question"]
    )

    llm = ChatOpenAI(
        temperature=settings["Temperature"],
        model=settings["Model"],
    )

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    # 创建文档组合链
    combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain)  # 添加 llm_chain 参数

    chain = ConversationalRetrievalChain(
        question_generator=llm_chain,
        retriever=ensemble_retriever,
        return_source_documents=True,
        memory=memory,
        combine_docs_chain=combine_docs_chain
    )

    msg.content = f'`{file.name}` 处理完成，请开始你的问答。'
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get('chain')

    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res['answer']
    source_documents = res['source_documents']

    text_elements = []
    if source_documents:
        for index, source_doc in enumerate(source_documents):
            source_name = f'来源{index + 1}'
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f'\n\n来源: {", ".join(source_names)}'
        else:
            answer += '\n\n来源未找到'

    await cl.Message(content=answer, elements=text_elements).send()