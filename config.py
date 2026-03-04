"""
Whisper 자막 생성 시스템 설정 파일
==================================

이 파일에서 모델 크기, 언어 설정, 폴더 경로 등을 자유롭게 수정할 수 있습니다.
수정한 후 프로그램을 다시 실행하면 설정이 반영됩니다.
"""

# --- 1. Whisper 모델 설정 ---
# 모델 크기 종류: 'tiny', 'base', 'small', 'medium', 'large-v3'
# - tiny / base: 속도가 매우 빠르지만 정확도가 낮음
# - small / medium: 속도와 정확도의 균형이 좋음 (기본값: medium)
# - large-v3: 가장 정확하지만 속도가 느리고 VRAM(그래픽카드 메모리)을 많이 사용함
MODEL_NAME = "medium"

# --- 2. 언어 설정 ---
SOURCE_LANGUAGE = "ja"  # 원본 영상 언어 (일본어: ja)
TARGET_LANGUAGE = "ko"  # 번역할 언어 (한국어: ko)

# --- 3. 번역 설정 ---
# 한 번에 번역할 문장 수 (기본값: 50)
# 너무 크면 Google 번역 API에서 거부될 수 있고, 너무 작으면 속도가 느려집니다.
TRANSLATION_BATCH_SIZE = 50

# 한 번에 보낼 최대 글자 수 (기본값: 4000)
# Google 번역 API의 안정적인 처리를 위한 안전장치입니다.
MAX_CHARS_PER_BATCH = 4000

# --- 4. 경로 설정 ---
# 각 폴더의 이름을 바꿀 수 있습니다. (기본값 추천)
INPUT_DIR = "input"   # 영상을 넣는 폴더
OUTPUT_DIR = "output" # 자막이 저장되는 폴더
WORK_DIR = "work"     # 임시 작업 파일이 저장되는 폴더

# --- 5. 기타 설정 ---
# 사용할 장치 설정 ('auto', 'cuda', 'cpu')
# - auto: GPU가 있으면 cuda, 없으면 cpu 자동 선택
# - cuda: NVIDIA 그래픽카드 강제 사용
# - cpu: CPU 강제 사용
DEVICE = "auto"

# 연산 정밀도 설정 ('float16', 'int8', 'float32')
# - float16: GPU에서 가장 빠름 (권장)
# - int8: CPU에서 효율적임
# - auto: 장치에 맞춰 자동 선택
COMPUTE_TYPE = "auto"

# 작업이 완료된 후 work 폴더의 임시 오디오 파일을 삭제할지 여부 (True: 삭제, False: 유지)
CLEANUP_TEMP_FILES = False

# VAD(Voice Activity Detection) 필터 활성화 여부
# True일 경우 목소리가 없는 구간을 자동으로 건너뛰어 속도가 빨라집니다.
USE_VAD_FILTER = True
