# Whisper Faster Subtitle Generator 🚀

일본어 영상을 입력받아 **일본어 자막(.srt)**과 **한국어 번역 자막(.srt)**을 자동으로 생성해주는 도구입니다.

## ✨ 주요 특징

- **DeepL 연동:** 구글 번역보다 훨씬 자연스러운 **DeepL 번역 엔진**을 공식 지원합니다.
- **보안 강화:** `.env` 파일을 활용한 API 키 보안 시스템이 적용되었습니다.
- **고성능 전처리:** 불필요한 오디오 분할 과정 없이 `faster-whisper`로 직접 처리합니다.
- **배치 자동화:** `input` 폴더에 여러 영상을 넣으면 한 번에 순차적으로 처리합니다.
- **간편한 설정:** `config.py` 파일 하나로 모든 설정을 제어할 수 있습니다.

## 🛠 사전 준비

1. **Python 3.8+**
2. **FFmpeg:** 오디오 추출을 위해 필수입니다.
3. **NVIDIA GPU & CUDA (권장):** 그래픽카드 가속을 원할 경우 필요합니다.

## 📦 설치 및 설정 방법 (처음 한 번만)

```bash
# 저장소 복제
git clone https://github.com/GHKIM235/01_Whisper_faster.git
cd 01_Whisper_faster

# 필수 패키지 설치
pip install -r video_subtitles/requirements.txt

# API 키 설정 (보안 필수!)
# 1. .env.example 파일을 .env로 이름을 바꿉니다.
# 2. .env 파일을 열고 DEEPL_API_KEY에 발급받은 키를 넣습니다.
cp .env.example .env
```

## 🚀 사용 방법

1. **`input`** 폴더에 영상 파일을 넣습니다.
2. 아래 명령어를 실행합니다.
   ```bash
   python video_subtitles/main.py
   ```
3. **`output`** 폴더에서 완성된 자막(.srt)을 확인합니다.

## ⚙️ 설정 변경 (`config.py`)

`config.py`에서 모델 크기, 언어 설정 등을 바꿀 수 있습니다.
- `TRANSLATION_ENGINE`: "deepl" 또는 "google" 중 선택 가능

---
**업데이트 날짜:** 2026년 3월 4일
**개발:** Gemini CLI & GHKIM235
