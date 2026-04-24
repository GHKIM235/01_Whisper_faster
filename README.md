# Whisper Faster Subtitle Generator 🚀 (v1.4.0)

일본어 영상을 입력받아 **일본어 자막(.srt)**과 **한국어 번역 자막(.srt)**을 고속으로 생성해주는 전문가용 도구입니다.

## ✨ 주요 특징 (Update v1.4.0)

- **자막 공백 복구 (Gap Rescue):** 자막 사이의 긴 공백(6초 이상)을 감지하여 누락된 대사를 고감도로 재분석하고 복원합니다.
- **직관적인 실행:** `--file` 옵션 없이도 파일이나 폴더 경로를 바로 입력하여 실행할 수 있습니다.
- **초정밀 싱크 (Word-Level Precision):** 단어 단위 타임스탬프를 분석하여 대사가 끝나는 시점에 자막이 즉시 닫히는 '칼싱크'를 구현했습니다.
- **번역 전용 모드 (`--translate-only`):** GPU 분석 없이 기존에 추출된 데이터만으로 한국어 번역만 빠르게 수행할 수 있습니다.
- **지능형 로그 및 백업:** 영상별 개별 로그를 생성하며, 작업물 수정 전 타임스탬프와 함께 자동 백업하여 데이터를 보호합니다.

## 🛠 사전 준비

1. **Python 3.10+**
2. **FFmpeg:** 오디오 추출 필수 도구.
3. **NVIDIA GPU (CUDA):** 최적의 성능을 위해 권장됩니다.

## 🚀 사용 방법

### 1단계: 기본 자막 생성
`input` 폴더에 영상을 넣거나 경로를 지정하여 실행합니다.
```bash
# input 폴더 전체 작업
python video_subtitles/main.py

# 특정 파일 또는 폴더만 작업 (NEW!)
python video_subtitles/main.py "C:\경로\영상.mp4"
```

### 2단계: 누락된 대사 복구 (Gap Rescue)
자막이 비어 있는 구간이 많을 때 사용하여 숨겨진 대사를 찾아냅니다.
```bash
python video_subtitles/main.py "영상경로.mp4" --rescue
```

### 3단계: 한국어 번역 자막 생성
분석 데이터(`_segments.json`)가 있다면 GPU 없이 번역만 가동합니다.
```bash
python video_subtitles/main.py --translate-only
```

## ⚙️ 설정 변경 (`config.py`)

- `ENABLE_GAP_RESCUE`: `True`로 설정하면 기본적으로 공백 복구 기능을 사용합니다.
- `SKIP_TRANSLATION`: `True`로 설정하면 기본적으로 번역 과정을 생략합니다.
- `DEFAULT_MODE`: "movie" (기본) 또는 "music" 중 선택 가능.
- `MODEL_NAME`: "large-v3" (최고 권장).

## 📈 성능 및 효율
- **번역 효율:** 대량 배칭을 통해 DeepL API 호출 횟수를 최소화했습니다.
- **데이터 안전:** 모든 작업물은 덮어쓰기 전 타임스탬프와 함께 자동 백업됩니다. 특히 `--rescue` 사용 시 원본 데이터를 강제 백업합니다.
- **로그 위치:** 모든 실행 기록은 `logs/` 폴더에서 영상 이름별로 확인 가능합니다.

---
**최종 업데이트:** 2026년 4월 24일
**개발:** Gemini CLI (Interactive AI Engineer)
