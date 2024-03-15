from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_community.document_loaders import WebBaseLoader,JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""  # Replace with your key

# https://python.langchain.com/docs/modules/data_connection/document_loaders/json

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('0/', glob="**/*.md")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
# prompt = ChatPromptTemplate(input_variables=['context', 'question'], messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \nQuestion: {question} \nContext: {context} \nAnswer:"))])
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-4", temperature=1)

# llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = rag_chain.invoke(message+"記得回答時只能用繁體中文回我，謝謝!")
    return gpt_response

gr.ChatInterface(predict).launch(server_name="0.0.0.0")
