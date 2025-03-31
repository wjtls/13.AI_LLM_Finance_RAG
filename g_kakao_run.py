import threading
import subprocess
from e_response_start import Response_class  # Response_class 임포트

import pyautogui
from pywinauto import application

#이미지 읽고 저장, 한글텍스트화
import pygetwindow as gw
import pyperclip
import time

class total_run_loop:
    def __init__(self):
        # Lock 객체 생성
        self.lock = threading.Lock()
        self.previous_question= False
        self.kakao_title_list= ['금융 AI채팅'] #찾아야할 카톡방

    def run_data_create(self):
        while True:
            with self.lock:  # 해당 함수 실행시 Lock을 획득하여 다른 스레드가 실행되지 못하게 함
                #import a_data_create #데이터 호출 시작
                pass
            time.sleep(3600)  # 60분 대기

    def run_response(self):
        response_c = Response_class()  # Response_class 인스턴스 생성
        Agent_class,RAG_model_engine = response_c.load_model()
        idx=0
        while True:
            new_question, is_new_question, past_question = self.check_for_new_question()
            if is_new_question and new_question != False:
                # run_response 메서드를 호출하여 질문을 전달
                new_question + '이전 대화 채팅 내용은 다음과같다. 대화 맥락을 참고해: ' + str(past_question)
                res=response_c.run_response(Agent_class,RAG_model_engine,new_question)
                self.chat_response_data(res)
            idx+=1
            time.sleep(10)  # 10초마다 질문 확인 (원하는 주기로 조정 가능)

    def kakao_title_search(self,title):
        try:
            win = gw.getWindowsWithTitle(title)[0]
            # 해당 타이틀이 있는걸 찾음
            sptr = 'False'  # False면 카톡 실행중

        except Exception:
            win = gw.getWindowsWithTitle('')[0]
            sptr = 'True'

            pass
        return win, sptr


    def check_kakao_talk(self,title): #카톡의 채팅방을 띄움
        print(title, '= 찾아야할 방이름')

        win, sptr = self.kakao_title_search(title)

        if sptr == 'False':
            print("카카오톡 실행 중")
            app = application.Application().connect(handle=win._hWnd)  # 연결
            chat_window = app.top_window()
            chat_window.set_focus()  # 창 포커스

            # 창 위치를 (0, 0)으로 설정 (왼쪽 상단)
            chat_window.move_window(x=0, y=0)

            # 입력창의 위치를 클릭 (예: x=100, y=500) - 실제 위치에 맞게 조정 필요
            pyautogui.click(350, 200)  # 카카오톡 입력창 좌표 클릭

        else:  # 카톡 실행되지 않았을경우 카톡 실행하고 메세지 보냄
            print("바탕화면에 띄워지지 않은 채팅방입니다.")  # 메세지박스 뜸

        win.activate()

    def chat_response_data(self, response):  # 응답을 카톡에 전송
        response = str(response).strip()

        # 클립보드에 문자열 복사
        pyperclip.copy(response)

        for title in self.kakao_title_list:
            self.check_kakao_talk(title)  # 창 띄우기

            # 잠시 대기하여 창이 활성화될 시간을 줍니다.
            time.sleep(1)

            # 엔터 키를 눌러서 채팅입력
            pyautogui.click(350, 200)  # 카카오톡 입력창 좌표 클릭
            time.sleep(0.1)

            pyautogui.press('enter')
            time.sleep(0.3)

            # 응답 텍스트 입력
            pyautogui.hotkey('ctrl', 'v')  # 클립보드에서 붙여넣음
            time.sleep(0.1)

            # 엔터 키를 눌러서 전송
            pyautogui.press('enter')

            # 다음 창으로 이동하기 전에 잠시 대기
            time.sleep(0.3)

    def check_for_new_question(self): #카톡에서 질문 체크/ 내용 긁어와서 전처리
        # 질문이 있는지 확인하는 로직
        # 카톡 내용을 복사해와서 대화내용중 맨앞에 조비스! 라는 단어가 있으면 질문으로 기입
        # 이전 질문을 저장해뒀다가 같은 질문이면 대답하지않음. 다른질문이어야 새질문으로 사용

        #카톡 채팅방 띄우기
        kakao_title= self.kakao_title_list

        for title in kakao_title:
            self.check_kakao_talk(title) #창 맨앞으로
            window = gw.getWindowsWithTitle(title)[0]
            left, top, width, height = window.left, window.top, window.width, window.height

            '''
            # 카톡 채팅창의 화면을 캡처
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            screenshot.save('question_screenshot.png')  # 임시로 저장
            screenshot = np.array(screenshot)
        
            # 이미지 전처리 (그레이스케일 변환)
            gray_image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

            # 이미지에서 텍스트 추출
            reader = easyocr.Reader(['ko'])  # 한국어 지원
            result = reader.readtext(gray_image)
            '''

            # Ctrl+A로 모든 텍스트 선택
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.5)  # 잠시 대기

            # Ctrl+C로 복사
            pyautogui.hotkey('ctrl', 'c')
            time.sleep(0.3)  # 잠시 대기

            # 클립보드에서 텍스트 가져오기
            copied_text = pyperclip.paste()
            copied_text = copied_text.split('\n')[-5:]
            print(copied_text)

        is_new_question=False
        question=False

        명령이름 = '!봇'

        # 질문 체크 로직
        lines =copied_text
        for line in lines:
            line = line.strip()  # 앞뒤 공백 제거
            if 명령이름 in line:  # 질문의 조건
                question_start_index = line.index(명령이름) + len(명령이름)  # 질문 시작 인덱스
                question = line[question_start_index:].strip()  # 질문 내용 추출

        # 이전 질문과 비교 (!조비스가 포함된 마지막 대화로만 확인함)

        if question != self.previous_question:
            self.previous_question = question  # 새로운 질문 저장
            is_new_question = True

        return self.previous_question, is_new_question, question


if __name__ == "__main__":
    # 스레드 생성
    class_run = total_run_loop()
    class_run.check_for_new_question()

    # 스레드 정의
    thread_a = threading.Thread(target=class_run.run_data_create)
    thread_b = threading.Thread(target=class_run.run_response)

    # 스레드 시작
    thread_a.start()
    thread_b.start()

    # 메인 스레드는 계속 실행 상태 유지
    thread_a.join()
    thread_b.join()