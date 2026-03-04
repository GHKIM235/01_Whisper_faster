# Whisper Faster Subtitle Generator 🚀 (v1.3.0)

일본어 영상을 입력받아 **일본어 자막(.srt)**과 **한국어 번역 자막(.srt)**을 고속으로 생성해주는 전문가용 도구입니다.

## ✨ 주요 특징 (Update v1.3.0)

- **초정밀 싱크 (Word-Level Precision):** 단어 단위 타임스탬프를 분석하여 대사가 끝나는 시점에 자막이 즉시 닫히는 '칼싱크'를 구현했습니다.
- **가독성 최적화:** 너무 잘게 쪼개지지 않도록 자연스러운 호흡(8초 룰, 1.2초 여백 허용)을 적용했습니다.
- **번역 전용 모드 (`--translate-only`):** GPU 분석 없이 기존에 추출된 데이터만으로 한국어 번역만 빠르게 수행할 수 있습니다.
- **강력한 GPU 가속:** RTX 2060 기준 **CUDA 활용률 85%**를 달성하여 2시간 영화를 단 몇 분 만에 처리합니다.
- **지능형 로그 시스템:** 영상별 개별 로그 파일을 생성하며, **DeepL API의 실시간 소모량**을 자동 기록합니다.
- **안정적인 윈도우 지원:** 인코딩 오류 없는 텍스트 UI와 자동 파일 백업 시스템을 제공합니다.

## 🛠 사전 준비

1. **Python 3.10+**
2. **FFmpeg:** 오디오 추출 필수 도구.
3. **NVIDIA GPU (CUDA):** 최적의 성능을 위해 권장됩니다.

## 🚀 사용 방법

### 1단계: 분석 및 일본어 자막 생성 (API 아끼기)
번역 없이 인식률과 싱크만 먼저 테스트하고 싶을 때 사용합니다.
```bash
python video_subtitles/main.py --mode movie --skip-translate
```

### 2단계: 한국어 번역 자막 생성 (최종 결과물)
분석 데이터(`_segments.json`)가 있다면 GPU 없이 번역만 가동합니다.
```bash
python video_subtitles/main.py --translate-only
```

### 기타 옵션
```bash
# 노래 가사 자막 생성 (가사 전용 최적화 모드)
python video_subtitles/main.py --mode music

# 특정 파일 하나만 처리
python video_subtitles/main.py --file input/VIDEO.mp4 --mode movie
```

## ⚙️ 설정 변경 (`config.py`)

- `SKIP_TRANSLATION`: `True`로 설정하면 기본적으로 번역 과정을 생략합니다.
- `DEFAULT_MODE`: "movie" (기본) 또는 "music" 중 선택 가능.
- `MODEL_NAME`: "large-v3" (최고 권장).

## 📈 성능 및 효율
- **번역 효율:** 대량 배칭을 통해 DeepL API 호출 횟수를 최소화했습니다.
- **데이터 안전:** 모든 작업물은 덮어쓰기 전 타임스탬프와 함께 자동 백업됩니다.
- **로그 위치:** 모든 실행 기록은 `logs/` 폴더에서 영상 이름별로 확인 가능합니다.

---
**최종 업데이트:** 2026년 3월 5일
**개발:** Gemini CLI (Interactive AI Engineer)
