import os

os.system('pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 '
          '--index-url https://download.pytorch.org/whl/cu118')
os.system('pip install -r requirements.txt')