import os
import sys
import pandas as pd
import json


# 툴
from langchain.tools import tool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.agent_toolkits import FileManagementToolkit #파일매니저
from langchain_community.tools.tavily_search import TavilySearchResults # tavily
from langchain_core.tools import Tool

#가공데이터
print(os.getcwd(), 'tool.py 현재 python 실행경로 확인')
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from a_News_FRED_data_api import a_data_vectorDB_upload
raw_data_class = a_data_vectorDB_upload

class Agentic_RAG: # 인터넷 정보 + 툴 사용 Agent
    def __init__(self):
        pass

class Agent_tools: # 에이전트가 사용할 툴
    def __init__(self):
        pass
    
    @tool
    def dalle_tool(self,query): #이미지 생성모델 (툴)
        """이미지 생성이 필요할시 사용"""
        dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)
        return dalle.run(query)
    
    @tool
    def filemanage_tool(self):
        """ 작업 디렉토리 경로 설정"""
        working_directory = "tmp"

        # 파일 관리 도구 생성(파일 쓰기, 읽기, 디렉토리 목록 조회)
        file_tools = FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["write_file", "read_file", "list_directory"],
        ).get_tools()
        return file_tools
    
    
    @tool
    def create_web_tools(self):
        """웹서치 할때 사용합니다"""
        search_res = TavilySearchResults(k=4)
        return search_res


    @tool
    def finnhub_eco_top_news(self, n: int = 3): # finhub 경제 top 뉴스 (경제 전반 뉴스)
        """
        Finnhub에서 최근 600분간의 미국 경제 지표 관련 주요 뉴스를 가져옵니다.

        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "finhub_json"  # 폴더에 포함된 이름
        contain_file_name = '경제top뉴스'
        date_col_name = 'published_time'  # 수정할 날짜데이터
        data_count = n if n is not None else 5 #데이터 카운트


        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data = pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data =pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)

        return df_data


    @tool
    def finnhub_market_news(self, n: int = 4): # 시장 top뉴스 (주식회사 관련 뉴스)
        """
        Finnhub에서 최근 600분간 미국 대형회사들의 관련 뉴스를 가져옵니다.

        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_finhub_json"  # 폴더에 포함된 이름
        contain_file_name = '시장top뉴스'
        date_col_name = "datetime"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            df_data = pd.DataFrame(recent_data)  # raw 데이터 테이블
            df_data = df_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            df_data = pd.DataFrame(recent_data)  # raw 데이터 테이블
            df_data = df_data.sort_values(by=date_col_name, ascending=False)
        return df_data


    @tool
    def finnhub_fin_top_news(self, n: int = 4): # 금융 top (주요 주식회사들의 주가 등락 해석,원인파악 뉴스)
        """
        Finnhub에서 최근 600분간 미국 대형회사들의 주가 등락(변동) 해석 뉴스를 가져옵니다.

        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_FMP_json"  # 폴더에 포함된 이름
        contain_file_name = '금융top뉴스'
        date_col_name = "date"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            df_data = pd.DataFrame(recent_data)  # raw 데이터 테이블
            df_data = df_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            df_data = pd.DataFrame(recent_data)  # raw 데이터 테이블
            df_data = df_data.sort_values(by=date_col_name, ascending=False)
        return df_data


    @tool
    def finnhub_exchange_news(self, n: int = 4): # 외환 top뉴스
        """
        Finnhub에서 최근 600분간 미국 외환 관련 뉴스를 가져옵니다.

        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_finhub_json"  # 폴더에 포함된 이름
        contain_file_name = '외환top뉴스'
        date_col_name = "datetime"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            df_data = pd.DataFrame(recent_data)  # raw 데이터 테이블
            df_data = df_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            df_data = pd.DataFrame(recent_data)  # raw 데이터 테이블
            df_data = df_data.sort_values(by=date_col_name, ascending=False)
        return df_data


    @tool
    def main_keyword_news(self, n: int = 7): # 시장 주요키워드로 검색한 뉴스
        """
        최근 미국시장을 움직이고있는 주요 키워드를 파악하여 미국 경제의 중장기적 흐름을 알수있는 뉴스를 가져오는 함수입니다.

        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_naver_news_json"  # 폴더에 포함된 이름
        contain_file_name = '키워드뉴스'
        date_col_name = "Date"  # 수정할 날짜데이터
        data_count = n

        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            file_path = base_dir + f'/{contain_folder_name}/{contain_file_name}.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                all_json_data = json.load(f)
            filtered_data = {key: value for key, value in all_json_data.items() if
                             not (isinstance(value, dict) and value.get("Type") == "뉴스 요약")}

            df = pd.DataFrame.from_dict(filtered_data, orient='index')

            # 2. Date 컬럼을 datetime 형식으로 변환
            df[date_col_name] = pd.to_datetime(df[date_col_name])

            # 3. 날짜 기준으로 정렬
            df_data = df.sort_values(by=date_col_name, ascending=True)

        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            file_path = base_dir+f'/{contain_folder_name}/{contain_file_name}.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                all_json_data= json.load(f)
            filtered_data = {key: value for key, value in all_json_data.items() if not (isinstance(value, dict) and value.get("Type") == "뉴스 요약")}

            df = pd.DataFrame.from_dict(filtered_data, orient='index')

            # 2. Date 컬럼을 datetime 형식으로 변환
            df[date_col_name] = pd.to_datetime(df[date_col_name])

            # 3. 날짜 기준으로 정렬
            df_data = df.sort_values(by=date_col_name, ascending=True)

        return df_data[-data_count:]


    @tool
    def IPO_calender(self, n: int = 3):  # IPO캘린더
        """
        최근 미국의 IPO 회사를 파악합니다
        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 3입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_finhub_json"  # 폴더에 포함된 이름
        contain_file_name = 'IPO_캘린더'
        date_col_name = "date"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data = pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data = pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        return df_data

    @tool
    def gain_calender(self, n: int = 3):  # 실적 캘린더
        """
        최근 미국의 실적발표 시간을 가져옵니다 (시간 데이터만 있음)
        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 3입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_finhub_json"  # 폴더에 포함된 이름
        contain_file_name = '실적_캘린더'
        date_col_name = "date"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data = pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data = pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        return df_data


    @tool
    def company_MNA_news(self, n: int = 5):  #인수합병뉴스
        """
        최근 미국의 인수합병 M&A 뉴스를 가져옵니다
        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_finhub_json"  # 폴더에 포함된 이름
        contain_file_name = '인수합병top뉴스'
        date_col_name = "datetime"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data = pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            all_json_data = load_class.load_json_file(base_dir, contain_folder_name, contain_file_name)
            recent_data = load_class.process_and_sort_data(all_json_data, date_col_name, data_count)
            recent_data =pd.DataFrame(recent_data)
            df_data = recent_data.sort_values(by=date_col_name, ascending=False)
        return df_data

    def create_tools(self):
        tools_func = [
            {
                'func': self.finnhub_eco_top_news,
                'name': 'us_economic_indicator_news',
                'description': '미국의 다양한 뉴스를 호출, 입력: 가져올 기사 수 (정수).'
            },
            {
                'func': self.finnhub_market_news,
                'name': 'us_company_news',
                'description': '미국에서 최근소식이 있는 회사의 뉴스를 호출, 입력: 가져올 기사 수 (정수).'
            },
            {
                'func': self.finnhub_fin_top_news,
                'name': 'us_company_stock_news',
                'description': '미국에서 최근소식이 있는 회사의 주가 등락에대한 설명 뉴스 호출, 입력: 가져올 기사 수 (정수).'
            },
            {
                'func': self.finnhub_exchange_news,
                'name': 'us_forex_news',
                'description': '미국의 외환 뉴스를 호출'
            },
            {
                'func': self.main_keyword_news,
                'name': 'current_us_market_news',
                'description': '현재 미국 중장기적 거시시장을 움직이는 주요 이슈를 호출, 입력: 가져올 기사 수 (정수).'
            }
        ]

        tools_list = []

        for tool in tools_func:
            tools_list.append(Tool(
                func=tool.get('func'),
                name=tool.get('name'),
                description=tool.get('description'),
                args_schema=None  # 필요에 따라 args_schema 정의
            ))
        return tools_list

    def test(self, n: int = 7):  # 시장 주요키워드로 검색한 뉴스
        """
        최근 미국시장을 움직이고있는 주요 키워드를 파악하여 미국 경제의 중장기적 흐름을 알수있는 뉴스를 가져오는 함수입니다.

        매개변수: n (int): 가져올 뉴스 항목의 수. 기본값은 5입니다.
        반환값: DataFrame : 최근 경제 뉴스 항목들의 데이터 프레임을 반환
        """
        contain_folder_name = "data_naver_news_json"  # 폴더에 포함된 이름
        contain_file_name = '키워드뉴스'
        date_col_name = "Date"  # 수정할 날짜데이터
        data_count = n

        load_class = raw_data_class.data_load()
        try:
            base_dir = "a_News_FRED_data_api/raw_data"
            file_path = base_dir + f'/{contain_folder_name}/{contain_file_name}.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                all_json_data = json.load(f)
            filtered_data = {key: value for key, value in all_json_data.items() if
                             not (isinstance(value, dict) and value.get("Type") == "뉴스 요약")}

            df = pd.DataFrame.from_dict(filtered_data, orient='index')

            # 2. Date 컬럼을 datetime 형식으로 변환
            df[date_col_name] = pd.to_datetime(df[date_col_name])

            # 3. 날짜 기준으로 정렬
            df_data = df.sort_values(by=date_col_name, ascending=True)
        except:
            base_dir = "../a_News_FRED_data_api/raw_data"
            file_path = base_dir+f'/{contain_folder_name}/{contain_file_name}.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                all_json_data= json.load(f)
            filtered_data = {key: value for key, value in all_json_data.items() if not (isinstance(value, dict) and value.get("Type") == "뉴스 요약")}

            df = pd.DataFrame.from_dict(filtered_data, orient='index')

            # 2. Date 컬럼을 datetime 형식으로 변환
            df[date_col_name] = pd.to_datetime(df[date_col_name])

            # 3. 날짜 기준으로 정렬
            df_data = df.sort_values(by=date_col_name, ascending=True)
        return df_data[-data_count:]


if __name__ == '__main__':
    # pandas 출력 옵션 설정
    pd.set_option('display.max_rows', None)  # 모든 행 표시
    pd.set_option('display.max_columns', None)  # 모든 열 표시
    pd.set_option('display.width', None)  # 너비 제한 해제
    pd.set_option('display.max_colwidth', None)  # 열 너비 제한 해제


    tool_class = Agent_tools()
    print(tool_class.test())
