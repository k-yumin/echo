import os
import streamlit as st
from gtts import gTTS


def main():
    # check if it is first run.
    if os.path.exists('first'):
        first()
        os.remove('first')

    # title
    st.title('에코')

    # tabs
    tab_titles = ['음성 생성', '모델 학습']
    voice_tab, model_tab = st.tabs(tab_titles)

    with voice_tab:
        st.header(tab_titles[0])
        models = ('junseo', 'jimin')  # TODO
        model_option = st.selectbox('모델을 선택해주세요.', models)
        text = st.text_input('텍스트를 입력해주세요.')

        if st.button('음성 생성하기'):
            st.write('음성 생성 중..')  # TODO
            inferred_audio = open('diff-svc/results/test_output.wav', 'rb')
            st.audio(inferred_audio.read(), format='audio/wav')
            st.success('음성 생성 완료!')

    with model_tab:
        st.header(tab_titles[1])
        uploader_msg = '학습을 위한 목소리 녹음 파일을 업로드해주세요. 총 분량이 1시간 이상이어야 합니다.'
        voice_files = st.file_uploader(uploader_msg, accept_multiple_files=True)


def first():
    first_msg = '프로그램을 실행하기 위한 환경을 준비 중입니다. 10분 이상 소요됩니다. 프로그램을 종료하지 마세요.'
    print(first_msg)

    os.system('pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 '
              '--index-url https://download.pytorch.org/whl/cu118')
    os.system('pip install -r requirements.txt')


def infer(text):
    tts = gTTS(text, lang='ko')
    tts.save('google_voice.wav')
    # TODO


if __name__ == '__main__':
    main()
