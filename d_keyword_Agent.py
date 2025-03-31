"""
현재 전세계 금융 시장을 움직이는 핵심 키워드 2개만 호출해
세계의 악재, 미국의 금융 이슈, 세계증시 향후 전망으로 키워드를 찾아줘

다른말은 하지말고 키워드만 2개 말해줘.
키워드는 , 로 구분해
"""

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex

from a_Env import Env
from d_Agent import Agent, Report_Agent
import time
import a_Agent_sys_prompt




is_db_data_load = False  # False: chunk 데이터를 폴더내 파일참고하여 새로 생성  /   True: 데이터 Qdrant DB에서 불러오기
Meta_keword_load = True  # False:메타값 새로 생성,    /    True: 불러오기  False: 메타 키워드 새로 생성
model_name = 'Corrective'

db_index_name ='findata'
db_index_url ='https://findata-3jxl645.svc.aped-4627-b74a.pinecone.io'

db_data_params=[is_db_data_load,db_index_name,db_index_url]

env_class = Env(db_data_params)  # 환경

# 데이터 생성, 노드생성, 벡터 인덱스 생성
Agent_class = Agent(Meta_keword_load, db_data_params)

# pinecone DB / Qdrant DB
storage_context, vector_store, pinecone_index = env_class.pinecone_db()

# DB에서 데이터 불러오기(키워드 모델용)
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)  # DB에서 불러오기


def Keword_search():

    sys_content = a_Agent_sys_prompt.keyword_prompt

    Settings.llm = OpenAI(model='gpt-4o-mini', temperature=0.1, top_p=0.4, system_prompt=sys_content)

    질문 = """
    미국 금융시장의 악재 및 위험 이슈, 미국 금융시장 호재 이슈가 뭐지?
    """

    if model_name=='Corrective':

        start = time.time()
        corrective_rag_pack = Agent_class.Corrective_keyword_model(vector_index,'keyword_model')
        response = corrective_rag_pack.query(질문)
        print('--------------------키워드 모델 응답-------------------------')
        print(response)
        print('---------------------------------------------')

        end = time.time()
        print('소모시간 :',end-start)

    response = [str(data).strip() for data in str(response).split(',')]
    return response