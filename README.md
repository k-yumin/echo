# 에코 (ECHO)
- 제24회 충북컴퓨터꿈나무축제 고등학교 공모전(SW제작) 부문 출품작

## 사전 준비

### 0. 최소사양 및 권장사양
||최소사양|권장사양|
|:-:|:-:|:-:|
|RAM|8GB|16GB|
|GPU|GeForce GTX 1050 Ti|GeForce RTX 2070|
|[VRAM](https://en.wikipedia.org/wiki/Video_random-access_memory)|4GB|8GB|

### 1. Python 3.9 설치
 - https://www.python.org/downloads/release/python-390/

### 2. CUDA Toolkit 11.8 설치
 - https://developer.nvidia.com/cuda-11-8-0-download-archive

### 3. FFmpeg 설치

#### Windows
  - https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-essentials.7z
  - 압축 해제 후 bin 폴더를 환경 변수에 추가

#### Linux
```bash
sudo apt update
sudo apt install ffmpeg
```

### 4. 현재 리포지토리 다운로드

#### Windows
 - https://github.com/k-yumin/echo/archive/refs/heads/main.zip

#### Linux
```bash
sudo apt install git
git clone https://github.com/k-yumin/echo.git
```

### 5. Pre-trained 모델 다운로드
 - [Hubert 체크포인트 다운로드](https://mega.nz/folder/AstwSTjC#GfRANHw8AuuNnveTEVcHdg)
 - 압축 해제 후 checkpoints 폴더를 diff-svc 폴더 안으로 이동

#### (Optional) GPU 메모리가 6GB 이상인 경우
 - [NSF-HiFiGAN 체크포인트 다운로드](https://github.com/MLo7Ghinsan/MLo7_Diff-SVC_models/releases/download/diff-svc-necessary-checkpoints/nsf_hifigan.zip)
 - 압축 해제 후 checkpoints 폴더 안으로 이동
