# Whisper Faster Subtitle Generator 🚀

일본어 영상을 입력받아 **일본어 자막(.srt)**과 **한국어 번역 자막(.srt)**을 자동으로 생성해주는 도구입니다. `faster-whisper`를 사용하여 기존 Whisper보다 훨씬 빠른 속도와 높은 효율성을 자랑합니다.

## ✨ 주요 특징

- **고성능 전처리:** 불필요한 오디오 분할 과정 없이 원본 데이터를 직접 처리하여 전처리 시간을 단축했습니다.
- **GPU 가속 (CUDA):** NVIDIA 그래픽카드를 활용한 하드웨어 가속을 지원합니다.
- **배치 자동화:** `input` 폴더에 여러 영상을 넣으면 한 번에 순차적으로 처리합니다.
- **안정적인 번역:** Google 번역 API의 글자 수 제한을 고려한 지능형 배칭 시스템이 적용되었습니다.
- **간편한 설정:** `config.py` 파일 하나로 모든 설정을 제어할 수 있습니다.

## 🛠 사전 준비

이 프로그램을 실행하기 위해서는 다음의 도구들이 설치되어 있어야 합니다.

1. **Python 3.8+**
2. **FFmpeg:** 오디오 추출을 위해 필수입니다. (시스템 환경 변수에 등록 권장)
3. **NVIDIA GPU & CUDA (선택 사항):** 그래픽카드 가속을 원할 경우 CUDA Toolkit과 cuDNN 설치가 필요합니다.

## 📦 설치 방법

```bash
# 저장소 복제
git clone https://github.com/GHKIM235/01_Whisper_faster.git
cd 01_Whisper_faster

# 필수 패키지 설치
pip install -r video_subtitles/requirements.txt
```

## 🚀 사용 방법

1. 프로젝트 루트 폴더에 있는 **`input`** 폴더에 자막을 만들 영상 파일(`.mp4`, `.mkv` 등)을 넣습니다.
2. 터미널(CMD)에서 아래 명령어를 실행합니다.
   ```bash
   python video_subtitles/main.py
   ```
3. 작업이 완료되면 **`output`** 폴더에서 완성된 자막 파일(.srt)을 확인합니다.

## ⚙️ 설정 변경 (`config.py`)

`config.py` 파일을 열어 다음 항목들을 자유롭게 수정할 수 있습니다.

- `MODEL_NAME`: Whisper 모델 크기 (`tiny`, `base`, `small`, `medium`, `large-v3`)
- `SOURCE_LANGUAGE`: 원본 영상 언어 (기본값: `ja`)
- `TARGET_LANGUAGE`: 번역될 언어 (기본값: `ko`)
- `DEVICE`: 실행 장치 (`auto`, `cuda`, `cpu`)
- `CLEANUP_TEMP_FILES`: 작업 완료 후 임시 파일 삭제 여부

## 📂 폴더 구조

- **`input/`**: 처리할 원본 영상 보관
- **`output/`**: 생성된 일본어/한국어 자막 및 데이터 보관
- **`work/`**: 작업 중 생성되는 임시 오디오 및 로그 데이터 보관
- **`video_subtitles/`**: 소스 코드 폴더

---
**개발 및 유지보수:** Gemini CLI & GHKIM235
