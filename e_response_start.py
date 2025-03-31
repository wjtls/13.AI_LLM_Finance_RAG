

# RAG: 리트리버 성능에 크게 의존, 왜 해당문서를 참조했는지 어떤부분이 잘못된것인지 불완전성고려 불가
# RAFT: 특정 도메인 파인튜닝 추가로 리트리버에 의존하지않고 모델 추론능력에 의존한다 = CoT 스타일 답변A으로 참조문서를 인용하게함 = 질문 + D*(질문과 관련있는 문서) + D(관련없는 문서들)-> Cot 스타일로 답변 생성
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 텍스트 벡터스토어
from llama_index.core import VectorStoreIndex
# 멀티모달 벡터스토어
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
import os
import sys
import asyncio
import json
from uuid import uuid4
import logging
from datetime import datetime
from dotenv import load_dotenv

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

from langchain_cohere import CohereRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from a_Agent_sys_prompt import sys_class

from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv(override=True)
# 시스템
import time
import os, sys
import pandas as pd
import importlib

# -----------------------랭체인

sys.path.append(os.getcwd())


from langchain_teddynote.community.pinecone import Pinecone
from langchain_teddynote import logging
from langchain.embeddings import OpenAIEmbeddings  # 임베딩 모델
from langchain_google_genai import ChatGoogleGenerativeAI

# 경로를 환경 변수에 추가
os.environ['PYTHONPATH'] = os.path.abspath('')
sys.path.append(os.environ['PYTHONPATH'])

# 랭스미스 추적
#logging.langsmith("hscho_AI")












# ---------------------------환경 세팅
# 환경 호출

import a_Env
from d_Agent import Result_Agent

import datetime
import a_market_condition
from c_Advance_method import AdvanceMethod
from d_langgraph_node import Graph_node as RAG_Graph

class Response_class:
    def __init__(self, meta_data_filter, QA_type):
        self.model_name = 'report'
        self.chunk_size = 300
        self.env_class = 0
        self.past_data = {}  # 과거 대화내역
        self.past_question = []
        self.past_answer = []
        self.meta_data_filter = meta_data_filter

        # db정보
        self.is_db_data_load = True
        self.db_index_name = 'finance-hscho'  # 인덱스 명
        self.db_index_url = f"https://{self.db_index_name}-3jxl645.svc.aped-4627-b74a.pinecone.io"
        self.db_data_params = [self.is_db_data_load,self.db_index_name, self.db_index_url]  # 사용 인덱스명,주소

        # 환경 호출
        self.env_class = a_Env.Env(self.db_data_params)  # 환경
        self.sys_prompt = sys_class() #프롬프트
        self.market_condition_class = a_market_condition.market_condition()
        #llm 정보
        self.llm_info = {
            "model": "gemini-2.0-flash", #"gemini-2.0-flash","gpt-4o"
            "temperature": 0.5,
            "max_output_tokens": 16384,
            "max_all_chunk_size": 128000,
            "context_window_size": 128000,
            "embedding_model": "text-embedding-3-large"
        }
        #최종응답, 중간 노드 llm
        self.node_llm = ChatGoogleGenerativeAI(model=self.llm_info.get('model', "gemini-2.0-flash"),
                                               temperature=self.llm_info.get('temperature', 0.2),
                                               max_output_tokens=8192,
                                               api_key=os.environ["GOOGLE_API_KEY"])  # 노드에서 사용되는 gpt(요약,웹서치 키워드생성,판단 등)

        #멀티쿼리 LLM
        #self.llm = ChatOpenAI(temperature=self.llm_info.get('temperature', 0.2),max_tokens=self.llm_info.get('max_output_tokens', 16384), model='gpt-4o')  # o3-mini,gpt-4o
        self.llm = ChatOpenAI(model ='gpt-4o-mini')

        #임베딩
        self.embeddings = OpenAIEmbeddings(model=self.llm_info.get('embedding_model', 'text-embedding-3-large'))

        #리트리버 클래스
        self.AdvanceMethod = AdvanceMethod(self.db_data_params, self.embeddings) #리트리버

        #기타
        self.QA_type=QA_type #출력타입



    def load_AI_trader_result(self): #AI trader 의 매매의사결정 내역
        data = []
        try:
            with open('b_strategy_AI/2_AI_APPO_LS_stock/traj/backtest_result.json', 'r') as file:
                for line in file:
                    data.append(json.loads(line.strip()))
        except:
            with open('../b_strategy_AI/2_AI_APPO_LS_stock/traj/backtest_result.json', 'r') as file:
                for line in file:
                    data.append(json.loads(line.strip()))

        # DataFrame으로 변환
        df = pd.DataFrame(data)

        # 필요한 데이터 추출
        def extract_data(column_name):
            data = df.loc[df['index'] == column_name, '0'].values[0]['long'][-10:]
            # 첫 번째 요소가 iterable한지 확인
            return [item[0] if hasattr(item, '__iter__') and not isinstance(item, str) else item for item in data]

        pv_log_return_data = extract_data('PV_log_return_data')  # PV의 로그수익률
        date_data = extract_data('date_data')
        action_data = extract_data('action_data')
        action_ratio = extract_data('buysell_ratio')
        price_data = extract_data('price_data')
        agent_stock = extract_data('agent_stock')

        if str(action_data[-1]) == '2':
            action_res = '현재 기술적분석 -> 매수'
        elif str(action_data[-1]) == '1':
            action_res = '현재 기술적분석 -> 관망'
        else:
            action_res = '현재 기술적분석 -> 매도'

        if str(action_data[-1]) == '1' and int(float(agent_stock[-1])) <=0 :  # 관망인데 보유수량이 0인경우-->매도
            action_res = '현재 기술적분석 -> 매도'

        result_ = f'매매의사결정: {action_res} ,\n 의사결정 시점:{date_data[-1]},\n 현재 NASQ 3배 보유 수량 {agent_stock[-1]},\n {date_data[-1]}시점 {action_res} 매매비중 : {action_ratio[-1]}, \n TQQQ 가격데이터(참고용이며 이야기하지마세요): {price_data[-1]},  \n매매의사결정 로그수익률(신뢰도 참고용, 이야기x) :{pv_log_return_data[-1]}'

        return result_

    def load_model(self):  # hscho = 문장분할 , hscho2 = 라마파서로 문장 분할 : 헤더가 들어가게끔
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%Y_%m_%d : %H:%M")
        print(f'AI 모델을 불러오고 있습니다... 현시각 : {now_time}')


        # pinecone DB / Qdrant DB
        vector_store, pinecone_index = self.env_class.pinecone_db(self.embeddings)

        # 에이전트 호출
        print('에이전트 호출 시작 ...')
        Agent = Result_Agent(self.llm)
        llm_retriever_Agent = Agent.Agentic_Agent(self.sys_prompt.report_retriever_prompt(now_time))  # 미래상황 예측 모델
        llm_realtime_Agent = Agent.Agentic_Agent(self.sys_prompt.report_realtime_data_prompt(now_time))  # 현상황 분석 모델

        # AI trader 결과
        AI_predict_result = self.load_AI_trader_result()
        print('AI 기술적분석 결과 :', AI_predict_result)

        # 리트리버 호출
        print('리트리버 호출 시작 ...')
        retriever = self.AdvanceMethod.create_retriever(pinecone_index,self.llm,self.sys_prompt.multi_query_prompt())

        # 그래프 생성
        Agent_node = RAG_Graph(llm_realtime_Agent,llm_retriever_Agent,self.node_llm,retriever,AI_predict_result).create_graph()
        print('그래프 생성 완료 ...')
        return Agent_node

    def save_response_to_json(self,response):# 응답저장 (절대경로로 설정, 외부에서 실행)
        file_path = "D:/AI_pycharm/pythonProject/3_AI_LLM_finance/b_finance_RAG_AI/traj/report_chat_history.json"

        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 파일이 존재하는지 확인
        if os.path.exists(file_path):
            # 기존 파일 읽기
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            # 파일이 없으면 새 리스트 생성
            data = []
        # 새 응답 추가
        data.append(response)
        # 전체 데이터를 파일에 쓰기
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    async def run_response(self,Agent_node,user_query):
        input_data = {
            "question": user_query,
        }
        response_chat = await Agent_node.ainvoke(input_data)
        response_chat = response_chat['answer']  # Graph내 answer 스테이트 반환값
        return response_chat

    async def report_response(self):
        ori_query = '미국증시 분석좀'

        # 시장상황 벡터 계산
        now_time = datetime.datetime.now()
        now_time = now_time.strftime("%Y-%m-%d %H:%M:%S")
        market_condition_value = self.market_condition_class.total_market_condition_index(
            str(now_time),  # 현시각
            window_size=60 * 24 * 30,
            n_splits=10,
            weight_dict={'nas': 0.4, 'bond': 0.4, 'money': 0.2}
        )

        #질문 재생성
        query = f'{ori_query} , 현재 시장상황 지표: {market_condition_value}'
        print('사용자 질문 입력 : ', query)

        # 에이전트 호출, 응답
        Agent_node = self.load_model()
        res = await self.run_response(Agent_node, query)
        response = {'datetime':now_time, '질문':ori_query, '시장상황':market_condition_value ,'응답':res}
        self.save_response_to_json(response) # chat history 저장
        return response


if __name__ =='__main__':
    meta_data_filter = [""] #필터안함
    QA_type ='report'

    res_class = Response_class(meta_data_filter,QA_type)
    res = asyncio.run(res_class.report_response())
    print(res)