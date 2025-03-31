
import pandas as pd
from llama_index.core import Settings
import os

pd.set_option('display.max_rows', None)  # 모든 행을 출력
pd.set_option('display.max_columns', None)  # 모든 열을 출력


# llm
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain_teddynote.messages import AgentStreamParser
from langchain_core.tools import Tool
# React
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

# llm 툴 사용
from langchain.tools.retriever import create_retriever_tool

# py 호출
from d_Agentic_tools import Agent_tools
from AI_langchain_method import messages

# 프롬프트 모음
import a_Agent_sys_prompt as sys_class
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

class Result_Agent:
    """응답모델"""
    def __init__(self ,llm):
        self.total_cost  = 0  # 총비용
        self.total_token = 0  # 총 토큰
        self.llm = llm
        self.store = {}
        self.tools =  Agent_tools().create_tools() #툴생성

    def Agentic_Agent(self,sys_content):
        # Prompt 정의

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_content.template),
                ("placeholder", "{chat_history}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )


        # 에이전트에 툴 장착
        Agentic_name = 'tool'
        agent = create_tool_calling_agent(self.llm, self.tools, prompt) # 툴과 프롬프트 결합
        agent_executor = self.create_executor(agent ,self.tools ,Agentic_name)

        # 과거 로그 , 메세지 기억 에이전트
        agent =  RunnableWithMessageHistory(
            agent_executor,
            # 대화 session_id
            self.create_chat_history,
            # 프롬프트의 질문이 입력되는 key: "input"
            input_messages_key="question",
            # 프롬프트의 메시지가 입력되는 key: "chat_history"
            history_messages_key="chat_history",
        )
        return agent  # 출력, 비용, 토큰


    def ReAct_agent(self):
        pass


    def create_chat_history(self ,session_ids)  :# 챗 히스토리
        # session_id 를 저장할 딕셔너리 생성
        if session_ids not in self.store:  # session_id 가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            self.store[session_ids] = ChatMessageHistory()
        return self.store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


    def create_executor(self ,agent ,tools ,Agentic_name): # 툴과 결합 , 로그를 출력
        if Agentic_name=='tool':
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose = 중간단계출력

        if Agentic_name=='ReAct':
            # 메모리 설정
            memory = MemorySaver()
            agent_executor = create_react_agent(agent, tools, checkpointer=memory)
        return agent_executor


    def create_retriever_tools(self ,retriever):
        retriever_tool = create_retriever_tool(
            retriever,
            name="retriever",  # 도구 이름
            description="일단 가장 먼저 호출하세요",  # 도구 설명
        )
        return retriever_tool


    def response(self ,agent_with_chat_history ,query ,report_id): # 답변 출력
        # 각 단계별 출력을 위한 파서 생성
        agent_stream_parser = AgentStreamParser()
        response = agent_with_chat_history.stream(
            {"input": query},
            config={"configurable": {"session_id": report_id}}, # 세션 ID 설정
        )
        for step in response:
            agent_stream_parser.process_agent_steps(step)


