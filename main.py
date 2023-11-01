import streamlit as st
from gtts import gTTS


def main():
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


def infer(text):
    tts = gTTS(text, lang='ko')
    tts.save('google_voice.wav')
    # TODO


if __name__ == '__main__':
    main()
