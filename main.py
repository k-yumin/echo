import streamlit as st
import numpy as np
import soundfile as sf
import shutil
import io
import subprocess
import os
from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from st_audiorec import st_audiorec
from gtts import gTTS

nsf_hifigan = False
config_path = ''

def main():
    # title
    st.title('에코')

    # tabs
    tab_titles = ['음성 생성', '모델 학습', '목소리 유사도 측정']
    voice_tab, model_tab, test_tab = st.tabs(tab_titles)

    # 음성 생성 탭
    with voice_tab:
        st.header(tab_titles[0])

        # 모델 선택
        model_list = os.listdir('diff-svc/checkpoints')
        if 'nsf_hifigan' in model_list:
            nsf_hifigan = True
        
        model_except = ['hubert', '0102_xiaoma_pe', 'nsf_hifigan', '0109_hifigan_bigpopcs_hop128']
        model_list = [model for model in model_list if model not in model_except]
        model_option = st.selectbox('모델을 선택해주세요.', model_list)

        # 텍스트 입력
        text = st.text_input('텍스트를 입력해주세요.')
        
        # 추론 시작 버튼
        if st.button('음성 생성하기'):
            st.write('음성 생성 중...')
            infer(model_option, text)

            # 결과 출력
            inferred_audio = open('diff-svc/results/speech.wav', 'rb')
            st.audio(inferred_audio.read(), format='audio/wav')
            st.success('음성 생성 완료!')

    # 모델 학습 탭
    with model_tab:
        st.header(tab_titles[1])

        # 모델 이름 입력
        model_name = st.text_input('모델 이름을 입력해주세요.')

        # 모델 업로드
        uploader_msg = '학습을 위한 목소리 녹음 파일을 업로드해주세요. 총 분량이 1시간 이상이어야 합니다.'
        voice_files = st.file_uploader(uploader_msg, accept_multiple_files=True)

        col1, col2 = st.columns([.25, 1])

        # 학습 시작 버튼
        with col1:
            if st.button('모델 학습하기'):
                st.write('모델 학습 중...')
                train(model_name, voice_files)
        # TensorBoard 열기 버튼
        with col2:
            if st.button('TensorBoard 열기'):
                open_tensorboard(model_name)

    # 목소리 유사도 측정 탭 (결과물 퀄리티를 확인하기 위함)
    with test_tab:

        st.write('자신의 목소리를 녹음하면, 최근 생성된 결과물과 목소리 유사도를 측정할 수 있습니다.')

        try:
            os.remove('workspace/my_voice.wav')
        except FileNotFoundError:
            pass
        
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            f = open("workspace/my_voice.wav", "bx")
            f.write(wav_audio_data)
            f.close()
        
        # 목소리 유사도 측정 버튼
        if st.button("목소리 유사도 측정하기"):
            st.write('유사도 측정 중...')
            similarity = 100 * compare_voices('workspace/my_voice.wav', 'diff-svc/results/speech.wav') 
        
            # 유사도 평가
            if similarity > 70:
                estimate = '동일 인물의 목소리일 가능성이 높습니다.'
            elif similarity > 60:
                estimate = '동일 인물의 목소리일 가능성이 있습니다.'
            else:
                estimate = '동일 인물의 목소리일 가능성이 낮습니다.'
        
            st.success(f"Similarity: {similarity:.2f}% ({estimate})")


# 음성 생성 함수
def infer(model_name, text):
    tts = gTTS(text, lang='ko')
    tts.save('diff-svc/raw/text.wav')

    try:
        os.remove('diff-svc/infer_.py')
    except FileNotFoundError:
        pass

    with open('diff-svc/infer.py', 'rt') as fin:
        with open('diff-svc/infer_.py', 'wt') as fout:
            for line in fin:
                fout.write(line.replace('test', model_name))

    os.chdir('diff-svc')
    os.system('python infer_.py')
    os.chdir('..')

# 모델 학습 함수
def train(model_name, voice_files):
    presets = f"""raw_data_dir: diff-svc/preprocess_out/final
        binary_data_dir: diff-svc/data/binary/{model_name}
        speaker_id: {model_name}
        work_dir: diff-svc/checkpoints/{model_name}
        max_sentences: 10
        use_amp: true"""
    
    config_type = 'config_nsf.yaml' if nsf_hifigan else 'config.yaml'
    
    src = 'workspace/'+config_type
    config_path = 'diff-svc/training/'+config_type
    shutil.copy(src, config_path)

    with open(config_path,'r+') as c:
        content = f.read()
        f.seek(0, 0)
        f.write(presets.rstrip('\r\n') + '\n' + content)

    os.chdir('diff-svc')
    os.system('python sep_wav.py')
    os.system(f'python preprocessing/binarize.py --config {config_path}')
    os.system(f'CUDA_VISIBLE_DEVICES=0 python run.py --config {config_path} --exp_name {model_name} --reset')


# TensorBoard 열기 함수
def open_tensorboard(model_name):
    subprocess.run(['tensorboard', '--load_fast=true', '--reload_interval=1', '--reload_multifile=true',
        f'--logdir=diff-svc/checkpoints/{model_name}/lightning_logs', '--port=6006'])


# 목소리 유사도 측정 함수
def compare_voices(file_path1, file_path2):
    wav_path1 = Path(file_path1)
    wav_path2 = Path(file_path2)

    wav1 = preprocess_wav(wav_path1)
    wav2 = preprocess_wav(wav_path2)

    encoder = VoiceEncoder()

    # 두 음성 파일을 벡터 형태로 변환하여 특징 추출
    embedding1 = encoder.embed_utterance(wav1)
    embedding2 = encoder.embed_utterance(wav2)

    # 코사인 유사도(cosine similarity)를 사용하여 두 음성 간의 유사도 측정
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return similarity


if __name__ == '__main__':
    main()
