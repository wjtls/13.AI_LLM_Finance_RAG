# 벡터 DB
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore  # 노드로 생성

# 라마인덱스 메타 필터
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
# 임베딩
from llama_index.embeddings.openai import OpenAIEmbedding


# BM25 (sparse)
from pinecone_text.sparse import BM25Encoder
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

# 쿼리 재작성
from llama_index.llms.openai import OpenAI

# 하이브리드 서치
from llama_index.core.query_engine import RetrieverQueryEngine  # 쿼리엔진으로 바꿔줌

# 리랭커
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 서브쿼리 엔진
from llama_index.core.query_engine import SubQuestionQueryEngine

# 다큐먼트
from llama_index.core import Document, VectorStoreIndex

# Hyde
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine



#---------------------시스템
import os
import json
from typing import List, Dict, Any

from pydantic import Field
from typing import Optional

#비동기 실행
import nest_asyncio
nest_asyncio.apply()
import asyncio

# 환경 호출
from a_Env import Env

#================================랭체인
import importlib
import sys
import os

sys.path.append(os.getcwd())
pinecone_control = importlib.import_module('AI_langchain_method.community.pinecone')

from langchain_core.vectorstores import VectorStoreRetriever,VectorStore
from langchain.prompts import ChatPromptTemplate # 프롬프트 생성
from langchain_core.runnables import RunnablePassthrough # 가상 질문 생성
from langchain_core.output_parsers import StrOutputParser # 아웃풋 파서
from langchain_openai import ChatOpenAI #오픈AI llm

from langchain_cohere import CohereRerank #리랭커

from langchain.retrievers.multi_query import MultiQueryRetriever #멀티쿼리 엔진
import os
import sys
import asyncio
import json
from uuid import uuid4
import logging
from datetime import datetime
from dotenv import load_dotenv

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast
from langchain_teddynote import logging as teddynote_logging

from langchain_cohere import CohereRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 문맥 압축 리트리버
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever  #문맥압축 리트리버
from langchain_text_splitters import RecursiveCharacterTextSplitter # 재귀 text split


# 랭체인 메타필터
from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)


# 환경 호출
import a_Env as a_Env
from a_Env import Env
from c_Fusion_Retriever import FusionRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever


from pinecone_text.sparse import BM25Encoder


class AdvanceMethod:
    def __init__(self, db_data_params, embedding_model):
        self.db_data_params = db_data_params
        self.is_db_data_load, self.db_index_name, self.db_index_url = db_data_params
        self.BM_db_data_params = [self.is_db_data_load, "bm25-" + self.db_index_name,f'https://"bm25-"+{self.db_index_name}-3jxl645.svc.aped-4627-b74a.pinecone.io']  # BM25용 파라미터

        # 임베딩
        self.embedding_model =embedding_model #시멘틱
        self.sparse_encoder = BM25Encoder().default() #스파스

        #리랭커
        self.reranker_compressor = CohereRerank(top_n=30, model="rerank-multilingual-v3.0")

    def create_retriever(self,pinecone_index, multi_query_agent,multi_query_prompt):

        # 스파스/dense 하이브리드
        hybrid_retrieve_engine = PineconeHybridSearchRetriever(
                                    embeddings=self.embedding_model,
                                    sparse_encoder=self.sparse_encoder,
                                    index=pinecone_index,
                                    top_k = 5,
                                    alpha=0.5
                                )
        multi_query_retriever = MultiQueryRetriever.from_llm(hybrid_retrieve_engine, multi_query_agent, prompt=multi_query_prompt)
        # self.multiquery_debug(multi_query_retriever,user_query) #멀티쿼리엔진 log확인

        # 문맥 압축 검색기 설정 - Reranker 설정.
        compression_retriever = ContextualCompressionRetriever(base_compressor=self.reranker_compressor, base_retriever=multi_query_retriever)
        return compression_retriever

    def multiquery_debug(self, retriever, query):  # 멀티쿼리 로그확인
        import logging

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        docs = retriever.invoke(query)
        docs_len = len(docs)
        for doc in docs:
            print(f'=============멀티쿼리엔진 출력 확인 , 총개수 : {docs_len} ================\n\n {doc.page_content}\n\n')


    def Sparse_search_BM25(self, doc):  # 파인콘 BM25 사용방식 (추가 수정 필요)
        print('pinecone BM25,Dense start')

        storage_context, vector_store, pinecone_index = Env(self.db_data_params).pinecone_db()  # 디비에서 호출한 벡터 인덱스

        report_id, report_type, custormer_report_name = self.meta_data_filter

        # 멀티 메타값 필터링 (OR)
        filter_data = []
        for id in report_id:
            filter_data.append(MetadataFilter(key="report_id", operator=FilterOperator.EQ, value=str(id)))

        llama_index_filter = MetadataFilters(
            filters=filter_data,
            condition=FilterCondition.OR,
        )

        # Dense 인덱스,리트리버 생성
        dense_vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        vector_retriever = dense_vector_index.as_retriever(similarity_top_k=10,
                                                           filters=llama_index_filter)  # 의미적유사도 리트리버

        # 스파스/dense 하이브리드
        hybrid_retrieve_engine = PineconeRetrieveEngine(pinecone_index, top_k=15,
                                                        filters=llama_index_filter)  # 여기서 topk는 리포트마다 k개만큼 청크리트리브

        return hybrid_retrieve_engine, vector_retriever

    def Sparse_search_BM25___(self, doc):  # BM25인덱스로 호출
        print('llama_index BM25,Dense start')
        # bm25 retriever (sparse search)
        # bm25는 임베딩 되지않은 doc을 사용해야한다. (헤더값과 마크다운만 적용된 doc임)

        # BM_storage_context, BM_vector_store,BM_pinecone_index = Env(self.BM_db_data_params).pinecone_db()
        storage_context, vector_store, pinecone_index = Env(self.db_data_params).pinecone_db()  # 디비에서 호출한 벡터 인덱스

        path = f'fip_api/ALI_QA_model_v1_hscho/traj/BM25_storage_{self.db_index_name}'  # ALI1 인덱스 공유
        if not os.path.exists(path):  # 폴더 생성
            os.makedirs(path)

        if self.is_db_data_load == False:
            sparse_vector_index = VectorStoreIndex.from_documents(
                doc)  # doc 을 바로 vector DB에 저장 (pinecone에 넣으면 인덱싱이 되므로)
            sparse_vector_index.storage_context.persist(persist_dir=path)  # 로컬 경로에 저장

        if self.is_db_data_load == True:
            BM_storage_context = StorageContext.from_defaults(persist_dir=path)
            sparse_vector_index = load_index_from_storage(BM_storage_context)

        report_id, report_type, custormer_report_name = self.meta_data_filter

        # 멀티 메타값 필터링 (OR)
        filter_data = []
        for id in report_id:
            filter_data.append(MetadataFilter(key="report_id", operator=FilterOperator.EQ, value=str(id)))

        llama_index_filter = MetadataFilters(
            filters=filter_data,
            condition=FilterCondition.OR,
        )

        # 스파스 리트리버
        bm25_retriever = BM25Retriever.from_defaults(docstore=sparse_vector_index.docstore, similarity_top_k=20)

        # Dense 인덱스
        dense_vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # Dense 리트리버
        vector_retriever = dense_vector_index.as_retriever(similarity_top_k=20,
                                                           filters=llama_index_filter)  # 의미적유사도 이므로 수치데이터 까지 리트리버 함
        return bm25_retriever, vector_retriever

    def hybrid_search(self, breaker_query, dense_engine, sparse_engine, reranker):
        # 쿼리 브레이커에 사용할 llm 정의
        llm = OpenAI(model="gpt-4o")
        h_retriever = FusionRetriever(llm, [dense_engine, sparse_engine], sys_query=breaker_query,
                                      meta_filter=self.meta_data_filter, similarity_top_k=20)
        retriever_engine = RetrieverQueryEngine(h_retriever, node_postprocessors=[reranker])
        # retriever_engine = retriever.configure(filter={'report_id':report_id})
        return retriever_engine

    def HyDe(self, query_engine):  # Hypothetical Document Embeddings
        # 할루시네이션 가상문서를 의도적으로 생성해서 청크와함께 사용 (맥락 파악 가능)
        # 유사도 검색 알고리즘(ANN)을 이용
        hyde_trans = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(query_engine, hyde_trans)
        return hyde_query_engine

    def RERANK(self):
        # cohere api 키
        api_key = a_Env.cohere_API
        # 리랭커 top_k 정의
        cohere_rerank = CohereRerank(api_key=api_key, top_n=15)
        return cohere_rerank

    def Sub_question(self, query_engine_tools):
        # 서브 퀘스천 쿼리 엔진
        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            use_async=True,
        )
        return query_engine

    def multi_query_engine(self, retriever_engine):
        # 멀티쿼리 엔진
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=retriever_engine, llm=llm_agent)
        return multi_query_retriever

    def RERANK(self):
        # cohere api 키
        api_key = a_Env.cohere_API
        # 리랭커 top_k 정의
        cohere_rerank = CohereRerank(model="rerank-mulitlingual-v3.0", api_key=api_key, top_n=20)
        return cohere_rerank

    def create_retriever2(self):  # 파인콘 BM25 사용방식 (추가 수정 필요)
        vector_store, pinecone_index = Env(self.db_data_params).pinecone_db()  # 디비에서 호출한 벡터 인덱스
        report_id, report_type = self.meta_data_filter

        # 멀티 메타값 필터링 (OR)
        filter_data = []
        for id in report_id:
            filter_data.append(MetadataFilter(key="report_id", operator=FilterOperator.EQ, value=str(id)))

        llama_index_filter = MetadataFilters(
            filters=filter_data,
            condition=FilterCondition.OR,
        )

        # Dense 인덱스,리트리버 생성
        dense_retriever = pinecone_index.as_retriever(similarity_top_k=10, filters=llama_index_filter)  # 의미적유사도 리트리버

        # 스파스/dense 하이브리드
        hybrid_retrieve_engine = PineconeRetrieveEngine(pinecone_index, top_k=15,filters=llama_index_filter)  # 여기서 topk는 리포트마다 k개만큼 청크리트리브

        return hybrid_retrieve_engine, dense_retriever







class PineconeRetrieveEngine:  # 파인콘 index.query를 리트리브엔진으로 변환
    def __init__(self, pinecone_index, top_k, filters):
        self.index = pinecone_index
        self.bm25 = BM25Encoder.default()
        self.top_k = top_k  # ex n이면 리포트필터마다 n개씩가져옴(리포트 5개면 topk*5개)
        self.filters = filters

    def get_embedding(self, text):
        model = OpenAIEmbedding(model_name="text-embedding-3-large")
        embedding = model.get_text_embedding(text)
        return embedding

    async def get_embedding_async(self, text):
        model = OpenAIEmbedding(model_name="text-embedding-3-large")
        embedding = await model.aget_text_embedding(text)
        return embedding

    def retrieve(self, query):
        # Dense vector 생성
        dense_vector = self.get_embedding(query)

        # Sparse vector 생성
        self.bm25.fit(query)
        sparse_vector = self.bm25.encode_queries(query)

        # LlamaIndex 필터를 Pinecone 필터로 변환
        pinecone_filter = {}
        retrieved_chunks = []

        if self.filters:
            for filter in self.filters.filters:
                if filter.operator == FilterOperator.EQ:
                    pinecone_filter[filter.key] = filter.value

                # Pinecone 쿼리 실행
                result = self.index.query(
                    vector=dense_vector,
                    sparse_vector=sparse_vector,
                    top_k=self.top_k,
                    include_metadata=True,
                    filter=pinecone_filter
                )
                # 결과 처리
                for match in result['matches']:
                    chunk = match.metadata.get('_node_content', '')  # 키 없으면 빈 문자열
                    chunk = json.loads(chunk)
                    score = match.score
                    retrieved_chunks.append({
                        'chunk': chunk,
                        'score': score
                    })

        return retrieved_chunks

    async def aretrieve(self, query):
        # Dense vector 생성
        dense_vector = await self.get_embedding_async(query)

        # Sparse vector 생성
        self.bm25.fit(query)
        sparse_vector = self.bm25.encode_queries(query)
        # LlamaIndex 필터를 Pinecone 필터로 변환
        pinecone_filter = {}
        retrieve_nodes = []
        if self.filters:
            for filter in self.filters.filters:  # 필터(리포트아이디) 각각 리트리브 실행
                pinecone_filter[filter.key] = filter.value
                # Pinecone 쿼리 실행
                result = await asyncio.to_thread(
                    self.index.query,
                    vector=dense_vector,
                    sparse_vector=sparse_vector,
                    top_k=self.top_k,
                    include_metadata=True,
                    filter=pinecone_filter
                )

                # 결과 처리
                for match in result['matches']:
                    chunk = match.metadata.get('_node_content', '')
                    chunk_data = json.loads(chunk)

                    # Node로 생성(TextNode 필수)
                    doc = TextNode(
                        **chunk_data
                    )
                    # Node에 스코어값 생성(일반 doc에서 지원 x)
                    node_with_score = NodeWithScore(
                        node=doc,
                        score=match.score
                    )

                    retrieve_nodes.append(node_with_score)

        return retrieve_nodes


class Document_to_node(Document):  # 스코어 추가된 노드생성
    # pydantic에서 추가 허용
    node: Optional['Document_to_node'] = Field(default=None)

    def __init__(self, *args, score=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._score = score

    def __repr__(self):  # 노드에 스코어 추가하여 반환환
        return f"DocumentWithSparseVector(score = {self.score}, text={self.text},metadata={self.metadata},id={self.id_}, embedding={self.embedding},sparse_values={self.sparse_values}), relationships ={self.relationships}, hash= {self.hash},text_template={self.text_template}, excluded_embed_metadata_keys={self.excluded_embed_metadata_keys}, excluded_llm_metadata_keys={self.excluded_llm_metadata_keys}, metadata_separator={self.metadata_separator},metadata_template={self.metadata_template}"

    @property
    def node(self):
        return self

    @property
    def score(self, ):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @classmethod
    def from_document(cls, document, score_value):
        # new_doc = Document_to_node(
        #    **document,
        #	score=score_value)

        new_doc = Document_to_node(
            **document,
            score=score_value
        )
        return new_doc













