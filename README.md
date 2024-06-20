# DDSP-SVC 학습 및 관련 내용 총정리

## 링크 및 참고 정리   
음성 변환 AI 모델인 DDSP-SVC을 실습하고 관련 내용을 정리하였습니다.   
본 내용은 아래 링크에 있는 사이트에 나와있는 내용을 기반으로 하였고 관련한 내용을 실습하며 겪은 오류의 해결 방안이나 순서등을 따로 정리한 내용이니 아래 링크의 사이트에 들어가 학습을 진행하여도 아무런 문제 없으며
정리한 순서 그대로 따라해도 무리없이 해당 모델을 사용 할 수 있게 만든 가이드이니 참고바랍니다.

[DDSP-SVC 공식 github](https://github.com/yxlllc/DDSP-SVC/blob/master/ko_README.md)   
[DDSP-SVC 개인정리 github](https://github.com/wlsdml1114/DDSP-SVC-KOR)  
[DIFF-SVC 개인정리 github](https://github.com/wlsdml1114/diff-svc?tab=readme-ov-file)   
[DIFF-SVC 개인정리 youtube](https://github.com/wlsdml1114/DDSP-SVC-KOR)

$\it{\large{\color{yellow}DIFF-SVC는 \ 데이터 \ 전처리용으로만 \ 사용합니다. }}$


## 1. 기본 프로그램 및 모델 다운로드
### 1-1. 아나콘다 설치

아나콘다 설치 링크 : [anaconda3](https://www.anaconda.com/download)  
아나콘다는 파이썬 패키지 관리 및 가상 환경 구축에 매우 편리한 환경 관리 도구 입니다.
### 1-2. ffmpeg 설치
ffmpeg 설치 링크 : [ffmpeg](https://www.gyan.dev/ffmpeg/builds/)

ffmpeg는 데이터 전처리 시 .mp4 혹은 다른 형태의 확장자를 .wav 형태로 바꿔주는 소프트웨어입니다.
diff-svc로 데이터 전처리 시 의존성이 매우 크기 때문에 필수 설치사항입니다.
만약 데이터 전처리를 diff-svc가 아닌 다른 형태로 진행하였다면 스킵해도 무방합니다
또한 설치 후 환경 변수 세팅이 필요합니다. (해당 내용은 아래 참고)  

- window키 => 시스템 환경 변수 편집 => 환경 변수 => 시스템 환경 변수(Path) => 압축 해제한 ffmpeg 파일의 bin 경로 입력



### 1-3. CUDA 11.7 설치
CUDA 11.7 설치 링크 : [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

본 실습은 CUDA 11.7 버전으로 진행하고 있습니다. pytorch 설치 시 CUDA 버전에 의존하니 꼭 11.7로 설치해 주세요.

### 1-4. DDSP-SVC 다운로드

DDSP-SVC 다운로드 링크 : [DDSP-SVC](https://github.com/wlsdml1114/DDSP-SVC-KOR)


아래 사진과 같이 직접 다운로드 받거나
```
git clone [link]
```
명령어를 통해 다운로드 받습니다.

      
<img width="1048" alt="git hub download guide" src="https://github.com/YooJeYong/DDSP-SVC/assets/170379560/fd5d1ab4-2945-41e2-bf02-17848da3594d">

### 1-5. pretrained 모델 다운로드

model1 : [hubert](https://oo.pe/https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)


├── DDSP-SVC  
│   ├── pretrain  
│   │   ├── hubert  

model2 : [hubert2](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)

├── DDSP-SVC  
│   ├── pretrain  
│   │   ├── hubert  

model3 : [nsf_hifigan](https://oo.pe/https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)

├── DDSP-SVC  
│   ├── pretrain  
│   │   ├── nsf_hifigan  



경로에 압축해제 한 파일을 넣어줍니다.

### 1-6. goldwave 설치

추후 음성 파일을 합치는 과정에서 사용 할 goldwave를 설치합니다.

goldwave 설치 링크 : [goldwave](https://goldwave.com/release.php)

## 2. DIFF-SVC로 DataSet 전처리 (학습 데이터가 있다면 Skip)

### 2-1. DIFF-SVC 다운로드
DIFF-SVC 다운로드 링크 : [DIFF-SVC](https://github.com/wlsdml1114/diff-svc?tab=readme-ov-file)

위의 링크에서 받은 파일을 압축 해제합니다.

### 2-2. 전처리 할 음성 파일 위치 옮기기
압축 해제한 DIFF-SVC 파일의 하위 폴더인 preprocess 파일에 전처리 할 음성 파일을 옮깁니다.

├── DDSP-SVC  
│   ├── preprocess  

### 2-3. 음성 파일 전처리 실행
```
window키 -> anaconda prompt 관리자 권한으로 실행 
```
anaconda prompt 실행 후 diff-svc 폴더로 이동합니다.

```
cd /path/to/project/diff-svc-main 
```

diff-svc 경로 이동 후 아래 명령어를 실행합니다.

```
python sep_wav.py
```
위의 명령어를 실행하고 작업이 끝났다면 아래 경로에서 10~15초 사이로 파일이 잘린지 확인 합니다.

├── diff-svc  
│   ├── preprocess_out  
│   │   ├── final  

위의 작업이 이상없이 완료 됐다면 아래 경로로 위의 경로의 파일들을 모두 옮겨줍니다.

├── DDSP-SVC  
│   ├── data  
│   │   ├── train  
│   │   │   ├── audio  

## 3. DDSP-SVC Python 및 Anaconda 가상환경 세팅

### 3-1. 콘솔 실행

```
window키 -> anaconda prompt 관리자 권한으로 실행
```
이하 명령어들은 모두 prompt 상에서 실행됩니다.


### 3-2. 프로젝트 폴더로 이동

```
cd /path/to/project/DDSP-SVC-master/
```
위의 경로는 프로젝트 파일을 압축 해제한 경로입니다. 

### 3-3. Anaconda 가상환성 생성 및 활성

```
conda create -n ddsp-svc python=3.8
# 가상 환경 생성 및 ddsp-svc라는 이름으로 python 3.8 버전을 설치한다는 명령어
conda activate ddsp-svc
# 가상환경으로 진입하는 명령어 이후 맨앞에 (base)였던 환경이 ddsp-svc(내가 생성한 가상 환경의 이름)으로 전환됩니다.
```

### 3-4. Pytorch 설치

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 3-5. Pytorch 설치 확인

```
python
import torch
print(torch.__version__)
```
위의 명령어를 타이핑 하였을때 Pytorch의 버전이 나온다면 설치 완료입니다.
이후 ctr+z혹은 exit()을 타이핑하여 python 환경을 종료합니다.

### 3-6. requirements 설치

```
pip install -r requirements.txt
```
오류 없이 설치되어야 하고 만약 오류가 생긴다면 가상 환경 삭제 후 3-1부터 다시 시작합니다.

* anaconda 관련 명령어 

```
# Anaconda 관련 명령어

# 활성화된 가상 환경을 비활성화 합니다.
conda deactivate

# 생성된 가상 환경 목록을 출력합니다.
conda env list

# 생선되어있는 가상 환경 목록 중 하나를 삭제합니다.
conda env remove -n "가상 환경 이름"

```

## 4. 데이터 전처리 및 학습 

### 4-1. 데이터 전처리  

├── DDSP-SVC-master  
│   ├── data  
│   │   ├── train  
│   │   │   ├── audio  

위의 경로에 10~15초 사이로 잘린 wav 파일이 있는지 확인하고 아래 명령어를 실행합니다.

```
python draw.py
```

위의 명령어를 실행하면 미리 잘라둔 파일 중 가장 상태가 좋은 파일 5~10개가 아래 경로로 추가됩니다. 만약 5개 이하로 생성된다면 명령어를 반복 실행 해주세요.

├── DDSP-SVC-master  
│   ├── data  
│   │   ├── val  
│   │   │   ├── audio  

위의 경로에 있는 음성 파일들은 이후 음성 학습 시 학습 정도를 파악하는 용도의 ref 파일임으로 원하는 다른 파일로 교체해도 무방합니다.

### 4-2. DDSP-SVC 내부 전처리

```
python preprocess.py -c configs/combsub.yaml
```
위의 명령어를 실행하여 ddsp-svc 내부 전처리를 진행합니다.
완료되면 아래 경로에 f0, units, volume이라는 새로운 폴더가 생깁니다.

├── DDSP-SVC-master  
│   ├── data  
│   │   ├── train  
│   │   │   ├── audio  

## 5. 학습 진행

### 5-1. 학습 진행 전 config file 세팅

아래 경로의 combsub 파일에 다음과 같은 내용을 붙여넣습니다.

34행의 num_workers는 cpu와 gpu 사용 비율이고, 35행의 batch_size는 한번에 학습할 사이즈(gpu vram 사용량)를 정하는 결정하는 값임으로

컴퓨터 사양에 맞게 수정해주세요.

├── DDSP-SVC-master  
│   ├── configs  
│   │   ├── combsub.txt  
```

data:
  f0_extractor: "harvest" # 'parselmouth', 'dio', 'harvest', 'crepe', 'rmvpe' or 'fcpe'
  f0_min: 65 # about C2
  f0_max: 800 # about G5
  sampling_rate: 44100
  block_size: 512 # Equal to hop_length
  duration: 2 # Audio duration during training, must be less than the duration of the shortest audio clip
  encoder: "contentvec768l12" # 'hubertsoft', 'hubertbase', 'hubertbase768', 'contentvec', 'contentvec768' or 'contentvec768l12' or 'cnhubertsoftfish'
  cnhubertsoft_gate: 10
  encoder_sample_rate: 16000
  encoder_hop_size: 320
  encoder_out_channels: 768 # 256 if using 'hubertsoft'
  encoder_ckpt: pretrain/hubert/checkpoint_best_legacy_500.pt
  train_path: data/train # Create a folder named "audio" under this path and put the audio clip in it
  valid_path: data/val # Create a folder named "audio" under this path and put the audio clip in it
  extensions: # List of extension included in the data collection
    - wav
model:
  type: "CombSubFast"
  win_length: 2048
  n_spk: 1 # max number of different speakers
enhancer:
  type: "nsf-hifigan"
  ckpt: "pretrain/nsf_hifigan/model"
loss:
  fft_min: 256
  fft_max: 2048
  n_scale: 4 # rss kernel numbers
device: cuda
env:
  expdir: exp/combsub-test
  gpu_id: 0
train:
  num_workers: 2 # If your cpu and gpu are both very strong, set to 0 may be faster!
  batch_size: 24
  cache_all_data: true # Save Internal-Memory or Graphics-Memory if it is false, but may be slow
  cache_device: "cuda" # Set to 'cuda' to cache the data into the Graphics-Memory, fastest speed for strong gpu
  cache_fp16: true
  epochs: 100000
  interval_log: 10
  interval_val: 2000
  lr: 0.0005
  weight_decay: 0
  save_opt: false

```

### 5-2. 학습 진행
```
python train.py -c configs/combsub.yaml
```
위의 명령어를 실행하여 학습을 시작합니다.
학습 파일은 아래 경로로 저장됩니다.

├── DDSP-SVC-master  
│   ├── exp  
│   │   ├── combsub-test  

기본 설정으로 2000스텝마다 저장되며, 작업 관리자를 실행하여 vram 사용량이 적다면
config.yaml의 batch_size를 적절히 조절하여 학습 속도를 조절하여 학습을 진행합니다.
중간에 학습을 그만두고 싶다면 ctrl+c로 학습을 종료 할 수 있으며 중간에 학습을 종료하더라도 마지막 저장된 checkpoint부터 시작하기 때문에 중간에 종료한 시점부터 다시 시작이 가능합니다.
DIFF-SVC와 다르게 DDSP-SVC는 학습 중간에 결과 확인이 가능한데 중간 학습 정도를 파악하고 싶다면 새로운 커맨드 창을 실행하여 (cmd,anaconda prompt 등) 아래 명령어를 실행합니다.
```
tensorboard --logdir="..\DDSP-SVC-KOR-master\exp\combsub-test\logs"
```
위의 명령어를 실행 한 후 [http://localhost:6006/](http://localhost:6006/)로 접속하면 손실율(loss)와 현재 학습 중인 데이터의 수준을 들어볼 수 있습니다.
손실율이 어느정도 안정되고 학습 결과가 제법 괜찮다면 ctrl+c로 학습을 종료합니다.

* 학습 단계에서도 batch num_workers와 batch_size를 설정 할 수 있는데 5-1에서 설정한 값이 전역 변수라면 아래 경로에서 지정하는 값은 지역 지역 변수입니다.
├── DDSP-SVC-master  
│   ├── exp  
│   │   ├── combsub-test
│   │   ├── config.txt

## 6. 결과물 출력
### 6-1. 결과물 출력 전처리
학습한 음성을 원하는 음악 파일의 보컬에 덧씌우기 위해서는 vocal(목소리)과 instrument(배경음)를 분리해야 하기 때문에 UVR5를 사용하여 덧씌울 음악 파일의 전처리를 진행합니다.


<img width="1048" alt="UVR5 다운로드" src="https://github.com/YooJeYong/DDSP-SVC/assets/170379560/380d3d7f-a0e7-448d-a136-0a65e1224c59">  

[UVR5 다운로드](https://github.com/Anjok07/ultimatevocalremovergui/releases/tag/v5.6)
해당 링크로 들어가 UVR5를 다운로드하고 설치까지 완료해줍니다.


[UVR5 알고리즘 랭킹 사이트](https://mvsep.com/quality_checker/synth_leaderboard)
해당 링크로 들어가면 가장 좋은 알고리즘 조합이 랭킹별로 정리 되어있습니다.  
위의 사이트를 참고하여 작업을 진행해도 좋습니다.

분리한 vocal 파일은 vocal.wav, instrument(배경음)은 instrumental.wav로 이름을 변경하고 

├── DDSP-SVC-master  
│   ├── exp  

위의 경로에 해당 파일 2개를 넣어준 뒤

```
python main.py -i "C:\DDSP-SVC\exp\vocal.wav" -m "C:\DDSP-SVC\exp\combsub-test\model_best.pt" -o "C:\DDSP-SVC\exp\vocal_trans.wav" -k 0 -id 1 -eak 0
```
명령어를 실행하면 마지막 checkpoint 기준으로 vocal부분이 덧씌워진 vocal_trans.wav가 출력됩니다.


![goldwave](https://github.com/YooJeYong/DDSP-SVC/assets/170379560/239b7731-147a-419c-bd17-66a355c14c22)

이후 goldwave 실행하고 vocal_trans.wav와 instrumental.wav 파일을 drag & drop하여 위의 이미지와 같이 띄운 후 vocal_trans를 클릭하여 선택하고 ctrl+c로 복사하고,
다시 instrumental를 클릭하여 선택하고 ctrl+m을 눌러 vocal_trans과 instrumental를 병합하고 저장하면 학습한 목소리와 배경음이 합쳐진 결과물을 얻을 수 있습니다.




