"""
Whisper 자막 생성 시스템 설정 파일
==================================

이 파일에서 모델 크기, 언어 설정, 폴더 경로 등을 자유롭게 수정할 수 있습니다.
수정한 후 프로그램을 다시 실행하면 설정이 반영됩니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (API 키 보안)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- 1. Whisper 모델 설정 ---
MODEL_NAME = "large-v3"

# --- 2. 언어 설정 ---
SOURCE_LANGUAGE = "ja"
TARGET_LANGUAGE = "ko"

# --- 3. 번역 설정 ---
TRANSLATION_ENGINE = "deepl"  # 'google', 'deepl'

# DeepL API 키 (환경 변수에서 안전하게 가져옴)
# 만약 .env 파일에 키가 없다면 수동으로 여기에 넣을 수도 있습니다.
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")

TRANSLATION_BATCH_SIZE = 50
MAX_CHARS_PER_BATCH = 4000
SKIP_TRANSLATION = False # True로 설정하면 번역 단계를 건너뜁니다 (API 절약용)

# --- 4. 경로 설정 ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
WORK_DIR = "work"

# --- 5. 기타 설정 ---
DEVICE = "auto"
# 'float16': 최고 정확도 (권장, VRAM 6GB 이상)
# 'int8_float16': 속도와 정확도 균형 (VRAM 부족 시)
# 'int8': 최적의 속도 (VRAM 매우 부족 시)
COMPUTE_TYPE = "float16" 

CLEANUP_TEMP_FILES = False
USE_VAD_FILTER = True # Movie 모드 기본값

# --- 6. 고급 오디오 전처리 ---
# True로 설정 시 배경음/음악을 제거하고 목소리만 추출하여 인식률을 높입니다.
# (처음 실행 시 분리 모델 다운로드로 인해 시간이 다소 소요될 수 있습니다.)
USE_VOCAL_SEPARATION = False 

# --- 7. 작업 모드 설정 ---
# 'movie': 영화/드라마/강의 등 대화 중심 (VAD 사용, 일반 번역)
# 'music': 노래 가사 중심 (VAD 미사용, 강력한 환각 방지, 가사 힌트 제공)
DEFAULT_MODE = "movie"
