import streamlit as st
import pandas as pd
import numpy as np

def main() :
    st.title('ECHO')
    
    tab_titles = ['모델 학습', '음성 생성하기']
    model_tab, voice_tab = st.tabs(tab_titles)
    
    with model_tab: # 모델 학습 탭
        st.header(tab_titles[0])
        uploaded_voice_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        
    with voice_tab: # 음성 생성 탭
        st.header(tab_titles[1])
        models = ('kyumin', 'jimin', 'taegyu')
        modeloption = st.selectbox('모델을 선택해주세요',
                                models)
    
        text_to_speech = st.text_input('텍스트를 입력해주세요')
    
        if st.button("음성 생성하기"):
            st.write("음성 생성 중..")
            # 음성 데이터 생성 함수
        
            # 요 밑에 2줄은 그냥 해놓은거, 모델 넣어서 같은 이름으로 해놓으면 됨
        
            created_audio_file = open('model_audio_file.mp3','rb')
            created_audio_bytes = created_audio_file.read()
    
            st.audio(created_audio_bytes, format = 'audio/mp3') #요거 확장자 바꾸는거 잊지 말고
        
            st.success("음성 생성에 성공하였습니다!")

if __name__ == '__main__' :
    main()