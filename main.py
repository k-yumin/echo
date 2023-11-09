import os
import shutil
import streamlit as st
from gtts import gTTS

nsf_hifigan = False
config_path = ''

def main():
    # title
    st.title('에코')

    # tabs
    tab_titles = ['음성 생성', '모델 학습']
    voice_tab, model_tab = st.tabs(tab_titles)

    # voice tab
    with voice_tab:
        st.header(tab_titles[0])

        # model select
        model_list = os.listdir('diff-svc/checkpoints')
        if 'nsf_hifigan' in model_list:
            nsf_hifigan = True
        
        model_except = ['hubert', '0102_xiaoma_pe', 'nsf_hifigan', '0109_hifigan_bigpopcs_hop128']
        model_list = [model for model in model_list if model not in model_except]
        model_option = st.selectbox('모델을 선택해주세요.', model_list)

        # text input
        text = st.text_input('텍스트를 입력해주세요.')
        
        # infer button
        if st.button('음성 생성하기'):
            st.write('음성 생성 중...')
            infer(model_option, text)

            # get result
            inferred_audio = open('diff-svc/results/speech.wav', 'rb') # TODO
            st.audio(inferred_audio.read(), format='audio/wav')
            st.success('음성 생성 완료!')

    # model tab
    with model_tab:
        st.header(tab_titles[1])

        # model name input
        model_name = st.text_input('모델 이름을 입력해주세요.')

        # model upload
        uploader_msg = '학습을 위한 목소리 녹음 파일을 업로드해주세요. 총 분량이 1시간 이상이어야 합니다.'
        voice_files = st.file_uploader(uploader_msg, accept_multiple_files=True)

        # train button
        if st.button('모델 학습하기'):
            st.write('모델 학습 중...')
            train(model_name, voice_files)


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


if __name__ == '__main__':
    main()
