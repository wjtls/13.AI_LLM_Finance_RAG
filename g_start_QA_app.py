


from streamlit.web import cli
if __name__ =='__main__':
    cli.main_run(['f_streamlit_app.py'])
    #cli.main_run(['확인 연습2.py'])

'''
# 배포 방법


#requirements 만들기
    pip install pipreqs
    pipreqs /path/to/your/project  #디텍토리 위치
    

# GCP 클라우드 https://console.cloud.google.com/projectselector2/home/dashboard?authuser=1&hl=ko 프로젝트 생성


# docker 파일 만들기 (Dockfile 이란 이름의 도커 파일 만들고 코드작성)
    FROM python:3.12

    # 작업 디렉토리 설정
    WORKDIR /app

    # 현재 디렉토리의 모든 파일을 컨테이너로 복사
    COPY . .

    # requirements.txt에 나열된 패키지 설치
    RUN pip install -r requirements.txt

    # Streamlit 실행
    CMD ["streamlit", "run", "f_streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]


# docker 빌드(이미지 빌드)
    데탑에서 docker 실행
    cmd에서
    cd D:\AI_pycharm\pythonProject\3.AI_LLM_finance\fin_model
    docker build -t gcr.io/peppy-ratio-398706/finance-app .   <--뒤에 띄어쓰고 점까지 붙여야함

# 이미지 확인
    docker images
    docker rmi gcr.io/peppy-ratio-398706/finance-app #이미지 삭제임

# docker 이미지 푸시
    gcloud auth login
    gcloud auth activate-service-account --key-file=peppy-ratio-398706-922a56826d44.json
    gcloud config set project peppy-ratio-398706
    docker push gcr.io/peppy-ratio-398706/finance-app   #****여기서 peppy-ratio-398706는 프로젝트 ID임 이름아님****  링크: https://console.cloud.google.com/iam-admin/settings?authuser=1&orgonly=true&project=peppy-ratio-398706&supportedpurview=organizationId


# 배포 실행 
    https://console.cloud.google.com/run/detail/us-central1/finance-app/revisions?project=peppy-ratio-398706&authuser=1&orgonly=true&supportedpurview=organizationId
    GCP에 가서 이미지(도커) 넣고 서버 실행
    
    
docker container prune  # 사용하지 않는 컨테이너 삭제
docker image prune  # 사용하지 않는 이미지 삭제
'''

