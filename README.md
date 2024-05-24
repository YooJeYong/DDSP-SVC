# DDSP-SVC 학습 및 관련 내용 총정리

## 링크 및 참고 정리
음성 변환 AI 모델인 DDSP-SVC을 실습하고 관련 내용을 정리하였습니다.   
본 내용은 아래 링크에 있는 사이트에 나와있는 내용을 기반으로 하였고 관련한 내용을 실습하며 겪은 오류의 해결 방안이나 순서등을 따로 정리한 내용이니 아래 링크의 사이트에 들어가 학습을 진행하여도 아무런 문제 없으며
정리한 순서 그대로 따라해도 무리없이 해당 모델을 사용 할 수 있게 만든 가이드이니 참고바랍니다.

[DDSP-SVC 공식 github](https://github.com/yxlllc/DDSP-SVC/blob/master/ko_README.md)   
[음성 AI 커뮤니티](https://arca.live/b/aispeech/74125759)   
[DIFF-SVC 개인정리 github](https://github.com/wlsdml1114/diff-svc)   
[DIFF-SVC 개인정리 youtube](https://www.youtube.com/watch?v=8hJ1Wullg_g)  

$\it{\large{\color{yellow}DIFF-SVC는 \ 데이터 \ 전처리용으로만 \ 사용합니다. }}$


## 1. 기본 프로그램 및 모델 다운로드
### 1-1 아나콘다 설치

아나콘다 설치 링크 : [anaconda3](https://www.anaconda.com/download)
```
아나콘다는 파이썬 패키지 관리 및 가상 환경 구축에 매우 편리한 환경 관리 도구 입니다.
```
### 1-2 ffmpeg 설치
ffmpeg 설치 링크 : [ffmpeg](https://www.gyan.dev/ffmpeg/builds/)
```
ffmpeg는 데이터 전처리 시 .mp4 혹은 다른 형태의 확장자를 .wav 형태로 바꿔주는 소프트웨어입니다.
diff-svc로 데이터 전처리 시 의존성이 매우 크기 때문에 필수 설치사항입니다.
만약 데이터 전처리를 diff-svc가 아닌 다른 형태로 진행하였다면 스킵해도 무방합니다
```
### 1-3 CUDA 11.7 설치
CUDA 11.7 설치 링크 : [CUDA 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
```
본 실습은 CUDA 11.7 버전으로 진행하고 있습니다. pytorch 및 DIFF-SVC, DDSP-SVC는 여러 의존성에 의존하기 떄문에 버전에 매우 민감하니 꼭 11.7 버전으로 설치하세요.
```
### 1-4 DIFF-SVC 및 DDSP-SVC 다운로드
DIFF-SVC 다운로드 링크 : [DIFF-SVC](https://github.com/prophesier/diff-svc)
DDSP-SVC 다운로드 링크 : [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)

```
아래 사진과 같이 직접 다운로드 받거나
git clone [link]
명령어를 통해 다운로드 받습니다.
```
      
<img width="1048" alt="git hub download guide" src="https://github.com/YooJeYong/DDSP-SVC/assets/170379560/fd5d1ab4-2945-41e2-bf02-17848da3594d">
### 1-5 pretrained 모델 다운로드
hubert : [hubert](https://oo.pe/https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
```
C:\DDSP-SVC\pretrain\hubert
경로에 압축한 파일을 넣어줍니다.
```
nsf_hifigan : [nsf_hifigan](https://oo.pe/https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)
```
C:\DDSP-SVC\pretrain\nsf_hifigan
경로에 압축해제 한 파일을 넣어줍니다.
```



