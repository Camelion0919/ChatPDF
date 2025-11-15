__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic import hub
import streamlit as st
import tempfile
import os
# from dotenv import load_dotenv
# load_dotenv() # 개인 key
# api_key = os.getenv("OPENAI_API_KEY")

# upload 된 file 불러오기
def pdf_to_document(uploaded_file):
    #임시폴더 생성
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_filepath, 'wb') as f:
        f.write(uploaded_file.getvalue())

    #업로드 된 파일을 document 객체로 get
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split
    return pages

#title
st.title('ChatPDF 🦖')
st.write('---')

# openapi_key 입력 받기
api_key = st.text_input('OpenAI key', type='password')
st.button('set')

if api_key:
    uploaded_file = st.file_uploader("Please upload the PDF file", type=['pdf'])
    st.write('---')
    if uploaded_file is not None:
        pages = pdf_to_document(uploaded_file)

        # text 분할
        text_splitter = RecursiveCharacterTextSplitter(
            #set a really small chunk size, just to show.
            chunk_size=100, # 각 chunk의 최대 길이
            chunk_overlap=20, # 인접한 chunk 사이의 중복 영역, 문장이 끊기는 문제를 해결 하기 위해 20글자 겹침
            length_function=len, # chunk 길이를 측정하는 함수
            is_separator_regex=False,  # 단순한 문자열로 해석
        )
        texts = text_splitter.split_documents(pages)

        # 임베딩 (OpenAi key 사용)
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

        # Chroma vector DB
        db = Chroma.from_documents(texts, embeddings_model)
        # 배포시---
        # import chromadb
        # chromadb.api.client.SharedSystemClient.clear_system_cache()
        #----   
        
        # 스트리밍 처리할 Handler 생성
        class StreamHandler(BaseCallbackHandler):
            def __init__(self, container, initial_text=""):
                self.container = container
                self.text=initial_text
            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text+=token
                self.container.markdown(self.text)


        st.header('Your question')
        question = st.text_input('input')

        if st.button('run'):
            with st.spinner("Wait for it...", show_time=True):
                llm = ChatOpenAI(api_key=api_key, temperature=0)

                # Chroma 백터 저장소에 대한 Retriever 인스턴스 생성
                retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(),
                                                                  llm=llm)

                #prompt template
                prompt = hub.pull('rlm/rag-prompt')

                #출력공간 확보 stream 부분
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box)
                generate_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, openai_api_key=api_key, streaming=True, callbacks=[stream_handler])
      
                #generate (검색결과 format)
                def format_docs(docs):
                    return '\n\n'.join(doc.page_content for doc in docs)

                rag_chain = (
                    {'context':retriever_from_llm | format_docs, "question":RunnablePassthrough()}
                    | prompt
                    | generate_llm
                    | llm
                    | StrOutputParser()
                )

                #question
                result = rag_chain.invoke(question)
                st.write(result)

    

