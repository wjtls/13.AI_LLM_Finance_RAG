from langchain.schema import Document
import os
import sys
import importlib
import datetime
import asyncio
from pydantic import BaseModel, Field 
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI,OpenAI

#웹검색
from AI_langchain_method.tools.tavily import TavilySearch
#from AI_langchain_method.tools.tavily import GoogleNews #구글뉴스
#from langchain.utilities import SearxSearchWrapper #일반검색
#from langchain.utilities import AlphaVantageAPIWrapper #금융지표,가격 검색
#from langchain.utilities import GoogleFinanceAPIWrapper # 구글금융지표,가격 검색
#from langchain.utilities import WikipediaAPIWrapper #위키피디아
#from langchain.utilities import ArxivAPIWrapper #논문검색


#그래프 생성
from langgraph.graph import END, StateGraph, START


#환경
sys.path.append(os.getcwd())
retriever_control = importlib.import_module('AI_langchain_method.retrievers')
from a_Agent_sys_prompt import sys_class

# 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "The question to answer"]
    answer: Annotated[str, "The generation from the LLM"]
    chunk_grade_result: Annotated[str, "As a result of evaluating whether the chunk is correct"]
    answer_grade_result: Annotated[str, "As a result of evaluating whether the answer is correct"]
    documents: Annotated[List[str], "The documents retrieved"] # 문서 저장
    web_documents: Annotated[List[str], "web search data"] # 웹정보 저장(종합)
    user_web_documents: Annotated[List[str], "user qestion web search data"]  # 사용자 질문 웹정보 저장(종합)
    issue_web_documents: Annotated[List[str], "issue web search data"]  # 주요이슈 웹정보 저장(종합)
    web_question: Annotated[List[str], "web search keyword"] #웹검색 키워드 저장
    multi_question: Annotated[List[str], "save multi query"] # 멀티쿼리 저장
    web_search_count : int  # 무한루프 방지
    rewrite_count : int


class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved document."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    """A binary score to determine the relevance of the retrieved document."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' 또는 'no' 또는 'numeric_incorrect' 또는 'web_incorrect'")




class Graph_node:
    def __init__(self,llm_realtime_Agent,llm_retriever_Agent,node_model,retriever,AI_predict_result):
        self.llm_realtime_Agent = llm_realtime_Agent # 현상황 분석 모델
        self.llm_retriever_Agent = llm_retriever_Agent #미래상황 예측 모델
        self.retriever =retriever
        self.node_llm = node_model
        self.trading_action = AI_predict_result


    def init_state(self,state:GraphState):
        return {
            "answer":"",
            "web_documents": [],
            "user_web_documents" :[],
            "issue_web_documents":[],
            "web_question": [],
            "web_search_count": 0,
            "rewrite_count": 0
        }
    '''
    def multiquery_debug(self,retriever,query): # 멀티쿼리 로그확인
        import logging

        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        docs = retriever.invoke(query)
        docs_len = len(docs)
        for doc in docs:
            print(f'=============멀티쿼리엔진 출력 확인 , 총개수 : {docs_len} ================\n\n {doc.page_content}\n\n')
    '''

    def multiquery_start(self, retriever, query):
        import logging

        logging.basicConfig()
        logger = logging.getLogger("langchain.retrievers.multi_query")
        logger.setLevel(logging.INFO)

        questions = []

        # 생성된 질문 파악
        original_log_info = logger.info

        def custom_log_info(msg, *args, **kwargs):
            if "Generated queries:" in msg:
                try:
                    # 쉼표로 분리하지 않고 전체 쿼리 문자열을 그대로 추가
                    queries = msg.split("Generated queries:")[1].strip()
                    questions.append(queries)
                except IndexError:
                    print("Error: Could not extract queries from log message.")
            original_log_info(msg, *args, **kwargs)

        logger.info = custom_log_info

        docs = retriever.invoke(query)
        docs_len = len(docs)

        print(f'=============멀티쿼리엔진 가동, 청크 총개수 : {docs_len} ================\n\n')

        print("멀티쿼리로 생성된 질문들:")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")

        return questions, docs

    
    # 리트리버
    def retrieve(self,state: GraphState):
        print("\n==== RETRIEVE 노드 실행 ====\n")
        question = state["question"]
        multi_question,documents = self.multiquery_start(self.retriever,question)
        # 멀티쿼리 질문저장
        
        print('리트리브 청크수:', len(documents))
        return {"documents": documents, "multi_question":multi_question}

    # 쿼리 재작성 노드
    def query_rewriter(self,state: GraphState):
        llm = self.node_llm
        # Query Rewrite 시스템 프롬프트
        system = """당신은 입력 질문을 최적화된 더 나은 버전으로 변환하는 질문 재작성 전문가입니다.
                 RAG 청크 검색을 위해. 입력을 보고 근본적인 의미 의도/평균에 대해 추론하여 한글로 작성하세요"""

        # 프롬프트 정의
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        # Question Re-writer 체인 초기화
        question_rewriter = re_write_prompt | llm | StrOutputParser()
        input_data = question_rewriter.invoke({'question':str(state['question'])})
        print(f'재 생성된 쿼리 : {input_data}')
        return {'question' : input_data , "rewrite_count" : state["rewrite_count"]+1}
    

    def grade_answer(self,state: GraphState):
        # 기준 1: 웹정보 내용이 반영됐는가 (웹 정보 평가 True인경우)
        # 기준 2: 리트리버 청크의 수치값들과 응답내 수치값 일치하는가
        # 기준 3: 해당되지 않는 청크를 가져왔는가 (grade_documents 노드에서 실행)
        print("\n====[응답 평가 노드 실행]====\n")
        documents = state["documents"] # 청크
        web_documents = state["web_documents"] # 웹정보
        answer = state['answer']

        # 필터링된 문서
        filtered_docs = []
        relevant_doc_count = 0

        # LLM 초기화
        llm = self.node_llm

        # GradeDocuments 데이터 모델을 사용하여 구조화된 출력을 생성
        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # 시스템 프롬프트 정의
        answer_sys_promt = """
            당신은 [LLM응답]이 청크와 관련있는지 판단하는 할루시네이션 검증 전문가 입니다.\n
            기준1. [web document]의 정보가 질문과 연관있고, [LLM응답]에 반영됐는지 확인
            
            기준1가 만족시 = yes
            기준1가 틀린경우 = no
            기준1 정보없을때 = 'numeric_incorrect'
            기준1 정보없을때 = 'web_incorrect'
            [retrieved document], [web document] 가 [LLM응답]과 관련있는지 여부를 나타내기 위해 바이너리 스코어 'yes' 또는 'no' 또는 'numeric_incorrect' 또는 'web_incorrect' 를 제공하세요
            """
        
        # 채팅 프롬프트 템플릿 생성
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", answer_sys_promt),
                ("human", "LLM응답: \n\n {LLM_answer} \n\n retrieved document: {document} \n\n web document: {web_doc}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader
        doc_content= [d.page_content if type(d)!=str else d for d in documents] if len(documents)!=0 else ''

        
        print('=============응답 최종 평가==============')
        # Question-Document 의 관련성 평가
        score = retrieval_grader.invoke({"LLM_answer": answer, "document": doc_content , 'web_doc': web_documents})
        grade = score.binary_score
        print(f'평가 결과 : {grade}')

        #웹서치 ->
        evaluate_chunk = grade 
        return {'answer':answer, "answer_grade_result": evaluate_chunk}


    # 문서 평가 노드(바이너리 스코어)
    def grade_documents(self, state: GraphState):
        question = state["multi_question"] # 질문
        documents = state["documents"] # 응답
        web_documents =state['web_documents'] #웹서치 정보

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M")

        # 필터링된 문서
        filtered_docs = []
        relevant_doc_count = 0

        # 필터링된 웹문서
        web_filtered_docs=[]

        # LLM 초기화
        llm = self.node_llm

        # GradeDocuments 데이터 모델을 사용하여 구조화된 출력을 생성하는 LLM
        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # 시스템 프롬프트 정의
        system = f"""
            Retrieved document 와 User question 들의 관련성을 평가하는 채점자입니다. \n
            최신정보 (반드시 현 시점 {now} 으로부터 2~3일이내) 데이터여야하고,
            문서에 질문맥락상 필요한 정보 or 추가정보인 경우 관련성이 있는 것으로 평가하세요.
            
            
            아래 순서로 평가합니다
            1. 먼저 질문들의 맥락을 파악하여 어떤 추가 질문이 나올수있는지 생각하세요
            2. 나올수 있는 추가질문들도 고려해서 문서내 정보중 필요하다 싶은 정보를 넓은범위로 포함합니다
            3. 문서가 질문맥락상 필요한지 여부를 나타내기 위해 바이너리 스코어 'yes' 또는 'no'를 제공하세요
            """

        # 채팅 프롬프트 템플릿 생성
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader



        for idx , d in enumerate(documents): # 질문과 리트리브 청크의 연관성 평가
            score = retrieval_grader.invoke({"question": question, "document": d if isinstance(d,str) else d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print(f"==== [{idx+1}번째 리트리브 청크 관련 여부 : {grade}] ====")
                # 관련 있는 문서를 filtered_docs 에 추가
                filtered_docs.append(Document(page_content =str(d)) if isinstance(d,str) else d)
                relevant_doc_count += 1
            else:
                print(f"==== [{idx+1}번째 리트리브 청크 관련 여부: No] ====")
                continue
        
        if web_documents =='zero_web_data': # 웹검색 데이터가 없으면
            evaluate_chunk = 'web_incorrect'
        else:
            web_doc = (
                web_documents.split('\n\n\n\n') if isinstance(web_documents, str)
                else [content for doc_list in web_documents for content in doc_list[0].page_content.split('\n\n\n\n')]
                if isinstance(web_documents, list)
                else web_documents.page_content.split('\n\n\n\n')
            )

            for idx, d in enumerate(web_doc):
                score = retrieval_grader.invoke({"question": question, "document": d})
                grade = score.binary_score

                if grade == "yes":
                    print(f"===={idx+1}번째 웹정보 관련 여부 : {grade}] ====")
                    # 관련 있는 문서를 filtered_docs 에 추가
                    web_filtered_docs.append(Document(page_content =str(d)) if isinstance(d,str) else d)
                    relevant_doc_count += 1
                else:
                    print(f"==== [{idx+1}번째 웹정보 관련 여부: No] ====")
                    continue
            evaluate_chunk = "doc_incorrect" if relevant_doc_count == 0 else "evaluation_pass"
        return {"documents": filtered_docs, "web_documents":web_filtered_docs , "chunk_grade_result": evaluate_chunk}

    def retry_tavily_search(self, query, result_len):  # 모든 타빌리 키 시도
        N = 6  # 사용할 API 키 수
        self.api_keys = [os.environ.get(f'TAVILY_API_KEY')] + [os.environ.get(f'TAVILY_API_KEY{i}') for i in
                                                               range(1, N + 1)]


        for idx, api_key in enumerate(self.api_keys):
            print(f"시도횟수:{idx+1} , 현재 루프에서 사용 중인 API 키: {api_key} ")
            self.api_key = api_key  # 현재 API 키를 self.api_key에 할당

            try:
                web_search_tool = TavilySearch(
                    max_results=result_len,
                    days=4,
                    search_depth="advanced",
                    topic="general",  # general: 뉴스, 경제, 학술, 블로그 등
                    include_raw_content=True,
                    include_answer=True,
                    api_key=self.api_key  # API 키를 설정
                )

                docs = web_search_tool.invoke({"query": query})
                print("웹 api 적용 완료")
                return docs  # 성공적으로 검색한 결과를 반환

            except Exception as e:
                print(f"API 키 {self.api_key}로 검색 실패: {e}")  # 실패 시 정확한 오류 메시지 출력
                continue  # 다음 API 키 시도

        raise Exception("모든 API 키로 검색에 실패했습니다.")




    # 주요 이슈기반 웹 검색 노드
    async def issue_web_search(self, state: GraphState):
        print("\n==== [WEB SEARCH 노드 실행] ====\n")

        ########################### 검색1 doc생성 웹검색(시장 주요키워드 파악,문서 저장)
        market_main_docs = self.retry_tavily_search('최근 미국 금융시장 이슈',result_len=4)


        # 검색 결과
        market_main_docs_str = "\n\n\n\n".join([str(d["content"]) for d in market_main_docs])
        market_main_docs = Document(page_content=str(market_main_docs_str))

        # 시장 주요 키워드 추출
        market_web_prompt = sys_class().keyword_prompt()
        market_web_chain = market_web_prompt|self.node_llm
        market_main_keyword = market_web_chain.invoke({"question": '현재 미국 금융시장을 움직이는 주요 이슈,경제지표가 뭐지? 총 3~4개만 구분자 콤마로 말해줘', "context": market_main_docs})
        print('issue_query 메인 키워드 추출: ',market_main_keyword.content)

        #결과 통합 후 업뎃
        market_main_docs_str = market_main_docs_str.split("\n\n\n\n")
        new_web_doc = "\n\n\n\n".join([f"현재 시장 주요 키워드:{str(market_main_keyword.content)}\n{str_chunk}" for str_chunk in market_main_docs_str])
        new_web_doc = Document(page_content=str(new_web_doc))
        print(' issue_web_search 청크수:', len(market_main_docs_str))
        state["issue_web_documents"].append(new_web_doc)
        return {"issue_web_documents":new_web_doc}









    # 사용자 질문 기반 웹 검색 노드
    async def user_web_search(self, state: GraphState):
        print("\n==== [WEB SEARCH 노드 실행] ====\n")
        question = state["question"]
        answer_data = state["answer"]
        document = state["documents"]
        web_search_keyword = state["web_question"]  # 웹 검색 키워드
        ############################# 검색 웹검색 툴


        #추가필요 :질문+doc
        keyword_base_prompt = """**역할**: 전문 웹 검색 키워드 생성기
                                    **목표**: 사용자 질문과 참고문서를 확인하여 해당내용에서 시장을 움직이는 주요 요소파악, 추가로 필요한 정보를 추측하여 구체적이고 효과적인 검색 키워드 1~3개 생성
                                    **규칙**:
                                    1. 질문의 핵심 개념을 분해 → [주제][행동][대상] 구조로 키워드 재구성
                                    2. 동의어/상위어/하위어 활용 (예: "트렌드" → "분석", "추이", "패턴")
                                    3. 검색 효율성을 위해 특수기호(#, "") 제거
                                    4. 단일 키워드보다 2~3단어 조합 우선 생성

                                    **추론과정**:
                                    1. 질문과 참고문서를 확인하여 연관있는 질문으로 재구성하세요
                                        ex) 우주에대해 알려줘 -> 1. 우주는 어떻게 생겼을까? , 2.우주의 방사선은 어떤종류가있을까? , 3.블랙홀이 뭘까?
                                    2. 해당 재구성 질문들에서 맥락을 파악하여 구체화 하세요
                                        ex) 1.빅뱅과 게이지대칭성 정보 , 2.뮤온 중성자 감마선의 인체영향 , 3.슈바르츠실트 반지름과 특이점  4.블랙홀과 고차원우주의 연관성
                                    3. 최종으로 해당 구체화 질문들을 검색에 용이하게끔 변형하세요
                                        ex) 빅뱅 게이지대칭성, 뮤온 중성자 감마선 인체영향 , 슈바르츠실트 반지름, 블랙홀 특이점 


                                    참고 문서 :{context}
                                    이전 AI응답:{answer}
                                    
                                    ##응답은 2개 까지만 생성하고 전략은 물어보지마세요##
                                    출력은 다른말 하지말고 3번에 해당하는 최종 구체화 질문들만 출력하면 됩니다
                                    ########## 출력형식(응답 키워드 구분자 콤마로 생성) : 응답1,응답2
                                    """

        if len(web_search_keyword)==0 : #첫 검색인경우
            keyword_sys_prompt = keyword_base_prompt
        else: #n번째 검색인경우
            keyword_sys_prompt = keyword_base_prompt + f"""\n ###특이사항 : 이전 검색질문 "{str(web_search_keyword)}"에서 검색에 실패했습니다. 해당 키워드들은 피해서 재생성하세요 AI 응답을 참고하여 나오지않은 구체적 정보를 알기위한 검색키워드를 파악하여 생성해주세요"""
                                
        
        # 웹검색 키워드 생성 모델
        keyword_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", keyword_sys_prompt),
                ("human", "사용자 질문: {question}"),
            ]
        )

        docs = self.stripDocuments(document)
        docs = '' #DB넣지않음
        keyword_model = self.node_llm
        keyword_chain = keyword_prompt|keyword_model
        web_keyword = keyword_chain.invoke({"question": question, "context":docs ,"answer": answer_data})




        ########################## 웹 검색 (키워드는 n개)
        web_results = ''
        for web_key in str(web_keyword.content).split(','):
            docs = self.retry_tavily_search(str(web_key),result_len=2)
            # 검색 결과
            result = "\n\n\n\n".join([d["content"] for d in docs])
            web_results += result
            print(f"user_query 웹검색 키워드 : {web_key}")
            print(f"user_query 웹검색 결과 개수 : {len(docs)}")

        if len(web_results) == 0: 
            web_results = 'zero_web_data'
        else:    
            web_results = Document(page_content=web_results)
        
        web_search_keyword.append(f'검색 키워드 : {web_keyword.content} ') #키워드 저장

        print('user_web_search 청크수:', len(docs))
        state["user_web_documents"].append(web_results)
        return {"user_web_documents":web_results,"web_search_count": state["web_search_count"] +1}




    #청크 확인 조건부 엣지
    def decide_to_chunk(self, state: GraphState):
        print("==== [조건부 결정 노드] ====")
        grade = state["chunk_grade_result"] #쿼리 재작성 OR 응답으로 이동
        
        max_depth = 1 # 최대 루프 깊이
        if state["web_search_count"]>=max_depth or state["rewrite_count"]>=max_depth:
            print("==== [DECISION: 응답 노드로 이동 -- 재귀 루프 한도 초과] ====")
            return "evaluation_pass"
        
        if grade == "doc_incorrect":
            # 정보 보강이 필요한 경우
            print(
                "==== [DECISION: 쿼리재작성을 실행] ===="
            )
            # 쿼리 재작성 노드로 라우팅
            return grade
        
        if grade == "evaluation_pass":
            # 관련 문서가 존재하므로 답변 생성 단계(generate) 로 진행
            print("==== [DECISION: 응답 노드로 이동] ====")
            return grade
        
        if grade == "web_incorrect":
            print("==== [DECISION: 웹검색을 재실시합니다] ====")
            return grade
        



    # 응답 확인 조건부 엣지
    def decide_to_answer(self, state: GraphState):
        print("==== [최종단계 : QA모델 응답에 대한 ACTION 결정 노드] ====")
        
        grade = state["answer_grade_result"]
        max_depth = 2

        if state["web_search_count"]>=max_depth or state["rewrite_count"]>=max_depth:
            print("==== [DECISION: 종료 -- 재귀 루프 한도 초과로 종료] ====")
            return "all_correct"
        
        if grade == "numeric_incorrect":
            # 할루시네이션 발생 : 수치값 미흡
            print(
                "==== [DECISION: 쿼리 재작성 실행  -- 할루시네이션 발생(수치값 미흡)] ===="
            )
            # 쿼리 재작성 노드로 라우팅
            return "numeric_incorrect"
        
        if grade == "web_incorrect":
            # 할루시네이션 발생 : 수치값 미흡
            print(
                "==== [DECISION: 웹 재검색 실행  -- 할루시네이션 발생(웹정보 미흡)] ===="
            )
            # 쿼리 재작성 노드로 라우팅
            return "web_incorrect"
        
        if grade =='no':
            print("=== [DECISION: 쿼리 재작성 실행 -- 할루시네이션 발생(수치값과 웹정보 모두 미흡)] ===")

            return "all_incorrect"

        if grade == "yes" :
            # 관련 문서가 존재하므로 답변 생성 단계(generate) 로 진행
            print("==== [DECISION: 종료 -- 응답에 할루시네이션이 없습니다.] ====")
            return "all_correct"
        
    
    def stripDocuments(self, documents) -> str:
        try:
            combined_document = ""
            index = 1
            for doc in documents:
                index += 1
                if isinstance(doc, Document):
                    combined_document +=  doc.replace("\\n", "\n") + "\n\n" if isinstance(doc,str) else doc.page_content.replace("\\n", "\n") + "\n\n"
                elif isinstance(doc, tuple) or isinstance(doc, list):
                    for tdoc in doc:
                        if isinstance(tdoc, Document):
                            combined_document += tdoc.replace("\\n", "\n") + "\n\n" if isinstance(tdoc,str) else tdoc.page_content.replace("\\n", "\n") + "\n\n"
            return combined_document
        except Exception as e:
            print(e)
        return ""

    # 답변 생성 노드
    def generate_answer(self, state: GraphState):
        now= datetime.datetime.now()
        now_date=now.strftime("%d-%m-%Y %H:%M")
        question =state["question"]
        docs = state["documents"]
        web_docs = state["web_documents"]


        print("\n==== 응답 생성 ====\n")
        print(f"사용자 질문 : {question}\n\n")
        print(f'리트리브 문서 수: {len(docs)}')
        print(f'리트리브 웹 문서 수 : {len(web_docs)}')


        # RAG 체인 구성
        Agentic_realtime_chain = (
                        {"context": RunnablePassthrough(), "question": RunnablePassthrough() ,"now_date" : RunnablePassthrough(),"trading_action": RunnablePassthrough()}
                        | self.llm_realtime_Agent
                        | RunnableLambda(lambda x: x.get("output", "") if isinstance(x, dict) else str(x))  # 딕셔너리에서 output 추출
                    )
        Agentic_predict_chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "now_date": RunnablePassthrough(),"trading_action": RunnablePassthrough()}
                | self.llm_retriever_Agent
                | RunnableLambda(lambda x: x.get("output", "") if isinstance(x, dict) else str(x))  # 딕셔너리에서 output 추출
        )

        # --------------문서, 웹 기반 응답 생성:
        docs = self.stripDocuments([docs])
        web_docs = self.stripDocuments([web_docs])
        print('doc 문서 :',docs)
        print('==========================\n\n\n\n\n\n\n\n')
        print('web_docs 문서 :',web_docs)
        web_response = Agentic_realtime_chain.invoke({"context": web_docs, "now_date": now_date, "question": str("해당 context 내용들을 토대로 분석해줘"),"trading_action": self.trading_action}, config={"configurable": {"session_id": '1131'}})
        docs_response= Agentic_predict_chain.invoke({"context": docs,"now_date": now_date, "question": str("해당 context 내용들을 토대로 분석해줘"),"trading_action":self.trading_action},config={"configurable": {"session_id": '1131'}})
        new_context = self.stripDocuments([[Document(page_content='title : 웹정보 + 실시간정보 + 기술적분석으로 현시장상태 분석 결과 \n'+str(web_response))] , [Document(page_content='title : 리트리브정보(이전상황) + 실시간정보 + 기술적분석으로 앞으로 시나리오 분석 결과 \n'+str(docs_response))]])

        
        # --------------최종응답:  프롬프트 변경필요, 랭그래프 전체적으로 비동기 방식 추가 필요

        final_llm = self.node_llm
        prompt = sys_class().final_res_prompt()
        final_llm_chain = (
                        {"context": RunnablePassthrough(), "question": RunnablePassthrough(),'trading_action': RunnablePassthrough()}
                        | prompt
                        | final_llm
                        | StrOutputParser()  # 딕셔너리에서 output 추출
                    )

        response = final_llm_chain.invoke({"context": new_context, "question": str(f"해당 context 내용들을 참고로 분석해줘") ,'trading_action':self.trading_action},config={"configurable": {"session_id": '1131'}})

        answer_col = ['현시장 상황 정리','전망','최종 정리']
        answer_data = [str(docs_response),str(web_response),str(response)]
        total_answer_data = [f"======================{answer_col[step]}======================\n\n {answer_data[step]} \n\n" for step in range(len(answer_col))]
        total_res="\n".join(total_answer_data)

        print(total_res, '최종 응답 결과')
        return {"answer": total_res}

    async def web_parallel_execution(self,state: GraphState): #병렬 웹실행후 결과 병합
        user_search_task = asyncio.create_task(self.user_web_search(state))
        issue_search_task = asyncio.create_task(self.issue_web_search(state))

        await asyncio.gather(user_search_task, issue_search_task) #웹정보 호출후 state업데이트
        state['web_documents'].append([state['issue_web_documents'],state['user_web_documents']])
        return {"web_documents":[state['issue_web_documents'],state['user_web_documents']]}



    def create_graph(self):
        # 그래프 상태 초기화
        workflow = StateGraph(GraphState)
        
        # 함수 호출
        GN  = self
        init_state= GN.init_state
        retrieve = GN.retrieve
        grade_documents = GN.grade_documents
        grade_answer = GN.grade_answer

        answer = GN.generate_answer
        query_rewrite = GN.query_rewriter
        user_web_search = GN.user_web_search
        issue_web_search = GN.issue_web_search
        web_parallel_execution =RunnableLambda(GN.web_parallel_execution)

        Evaluation_chunk  = GN.decide_to_chunk # 조건부 엣지
        Evaluation_answer = GN.decide_to_answer # 조건부 엣지

        # ----------노드 정의------------
        workflow.add_node('init_state',init_state)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("LLM_answer", answer)
        workflow.add_node("query_rewrite", query_rewrite)
        #workflow.add_node("user_web_search_model", user_web_search)
        #workflow.add_node("issue_web_search_model", issue_web_search)
        workflow.add_node('grade_answer',grade_answer)
        workflow.add_node("web_parallel_group", web_parallel_execution)
        '''
        # ----------엣지 연결----------
        workflow.add_edge(START, "init_state")
        workflow.add_edge("init_state", "retrieve")
        workflow.add_edge("retrieve","issue_web_search_model")
        workflow.add_edge("issue_web_search_model","user_web_search")
        workflow.add_edge("user_web_search", "grade_documents")
        '''


        # 그룹 노드를 워크플로우에 추가
        workflow.add_edge(START, "init_state")
        workflow.add_edge("init_state", "retrieve")
        workflow.add_edge("retrieve", "web_parallel_group")
        workflow.add_edge("web_parallel_group","grade_documents")

        # 청크 평가 조건부 엣지 (할루시네이션 방지)
        workflow.add_conditional_edges(
            "grade_documents",  # 청크 평가후 decide to chunk노드로 이동
            Evaluation_chunk,
            {
                "doc_incorrect": "query_rewrite",
                "evaluation_pass": "LLM_answer",
                "web_incorrect": "web_parallel_group"
            },
        )

        workflow.add_edge("query_rewrite", "retrieve") # 쿼리 재작성으로 되돌아가는 겨웅
        workflow.add_edge("web_parallel_group", "grade_documents") #웹 평가로 되돌아간다
        workflow.add_edge("grade_documents", "LLM_answer") # 청크 판단후 응답
        workflow.add_edge("LLM_answer", "grade_answer")  # 응답 생성후 응답 평가
        workflow.add_conditional_edges( 
            "grade_answer",
            Evaluation_answer,
            {
                "web_incorrect":"web_parallel_group",
                "numeric_incorrect":"query_rewrite",
                "all_incorrect": "query_rewrite",
                "all_correct": END, 
            },
        )

        # 그래프 컴파일
        res_app = workflow.compile()
        print('===그래프 컴파일 완료===')

        #시각화
        graph_control = importlib.import_module('AI_langchain_method.graphs')
        graph_control.visualize_graph(res_app) # 시각화

        return res_app























