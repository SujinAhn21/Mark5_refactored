# robust_preprocess_audio.py
'''교수님 실제 코드에는 이거 안 쓰고 mark2 시리즈에서부터 사용했던 fix_audio_length_to_24000.py를 사용했습니다.
   파이프라인에 안넣었기 때문에 무시하시면 됩니다.'''

import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import traceback

# ==============================================================================
# 설정 (Configuration) - 여기를 수정하여 동작을 변경할 수 있습니다.
# ==============================================================================

# --- 폴더 경로 ---
INPUT_DIR = "data_source_test"
OUTPUT_DIR = "data_test"

# --- 오디오 설정 ---
# 목표 샘플링 레이트 (Target Sample Rate)
# 음성 데이터는 보통 16000, 22050, 24000, 48000 등을 사용합니다.
TARGET_SR = 48000

# 목표 오디오 길이 (샘플 수 기준)
# 48000 SR에서 5초 = 240000 샘플 (48000 * 5)
# 24000 SR에서 10초 = 240000 샘플 (24000 * 10)
TARGET_LENGTH_SAMPLES = 240000

# ==============================================================================

def process_audio_file(input_path, output_path, target_sr, target_len_samples):
    """
    단일 오디오 파일을 불러와서 전처리하고 저장합니다.
    """
    try:
        # 1. 오디오 파일 로드 및 리샘플링
        # mono=True로 설정하여 강제로 단일 채널(모노)로 변환합니다.
        wav, sr = librosa.load(input_path, sr=target_sr, mono=True)

        # 2. 오디오 길이 맞추기 (패딩 또는 자르기)
        current_len = len(wav)
        
        if current_len < target_len_samples:
            # 길이가 짧으면 뒤쪽에 0으로 패딩(padding)
            padding_needed = target_len_samples - current_len
            wav = np.pad(wav, (0, padding_needed), 'constant')
        elif current_len > target_len_samples:
            # 길이가 길면 앞에서부터 자르기(truncating)
            wav = wav[:target_len_samples]

        # 3. 전처리된 오디오 파일 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, target_sr)
        
        return True, None # 성공 시

    except Exception as e:
        # 파일 처리 중 오류 발생 시, 상세한 오류 메시지를 반환합니다.
        error_message = f"{type(e).__name__}: {str(e)}"
        return False, error_message


def run_preprocessing():
    """메인 실행 함수"""
    print("=" * 50)
    print("오디오 길이 정규화 시작...")
    print(f"입력 폴더: {INPUT_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"목표 샘플링 레이트: {TARGET_SR} Hz")
    print(f"목표 오디오 길이: {TARGET_LENGTH_SAMPLES} 샘플 ({TARGET_LENGTH_SAMPLES / TARGET_SR:.2f}초)")
    print("=" * 50)

    # 입력 폴더에서 모든 오디오 파일 (.wav, .mp3, .m4a 등) 검색
    audio_files = []
    supported_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(supported_extensions):
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"[경고] '{INPUT_DIR}' 폴더에서 처리할 오디오 파일을 찾을 수 없습니다.")
        return

    success_count = 0
    failed_files = []

    # tqdm을 사용해 진행 상황을 시각적으로 표시
    for input_path in tqdm(audio_files, desc="오디오 파일 처리 중"):
        relative_path = os.path.relpath(input_path, INPUT_DIR)
        # 원본 파일 확장자와 상관없이 .wav로 저장하도록 경로 수정
        output_relative_path = os.path.splitext(relative_path)[0] + '.wav'
        output_path = os.path.join(OUTPUT_DIR, output_relative_path)

        success, error_msg = process_audio_file(
            input_path, output_path, TARGET_SR, TARGET_LENGTH_SAMPLES
        )

        if success:
            success_count += 1
        else:
            failed_files.append((input_path, error_msg))

    # --- 최종 결과 요약 ---
    print("\n" + "=" * 20 + " 결과 요약 " + "=" * 20)
    print(f"총 {len(audio_files)}개 파일 스캔 완료.")
    print(f"성공적으로 처리된 파일: {success_count}개")
    print(f"처리 실패한 파일: {len(failed_files)}개")
    print(f"처리된 데이터 저장 위치: {OUTPUT_DIR}")
    print("=" * 51)
    
    if failed_files:
        print("\n[상세 실패 내역]")
        for path, reason in failed_files:
            print(f"  - 파일: {path}")
            print(f"    └─ 원인: {reason}")
            
# 스크립트 실행
run_preprocessing()  