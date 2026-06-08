# fix_audio_length_to_240000.py (리팩토링 버전)

import os
import glob
import librosa
import soundfile as sf
import numpy as np
import argparse
from tqdm import tqdm

TARGET_SR = 16000
TARGET_SAMPLES = 240000 # 15초 * 16000Hz

def process_audio_file(input_path, output_path):
    """오디오 파일을 로드하고 길이를 조절하여 저장합니다."""
    try:
        # librosa로 로드하면 자동으로 resample됨
        wav, sr = librosa.load(input_path, sr=TARGET_SR)

        if len(wav) > TARGET_SAMPLES:
            # 길면 자르기
            wav = wav[:TARGET_SAMPLES]
        elif len(wav) < TARGET_SAMPLES:
            # 짧으면 0으로 채우기 (패딩)
            padding = np.zeros(TARGET_SAMPLES - len(wav), dtype=np.float32)
            wav = np.concatenate([wav, padding])

        # 출력 폴더가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, wav, TARGET_SR, 'PCM_16')
        return True
    except Exception as e:
        print(f"[ERROR] 파일 처리 실패: {input_path} -> {e}")
        return False

def main(input_dir, output_dir):
    print("="*50)
    print("오디오 길이 정규화 시작...")
    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}")
    print("="*50)
    
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 입력 폴더를 찾을 수 없습니다: {input_dir}")
        return

    # 입력 폴더 내의 모든 .wav 파일을 재귀적으로 찾음
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    
    if not audio_files:
        print("[WARNING] 처리할 .wav 파일이 없습니다.")
        return

    success_count = 0
    for file_path in tqdm(audio_files, desc="오디오 파일 처리 중"):
        # 입력 폴더 경로를 제외한 상대 경로를 구함
        relative_path = os.path.relpath(file_path, input_dir)
        # 출력 경로를 만듦 (하위 폴더 구조 유지)
        output_file_path = os.path.join(output_dir, relative_path)
        
        if process_audio_file(file_path, output_file_path):
            success_count += 1
            
    print("\n--- 결과 요약 ---")
    print(f"총 {len(audio_files)}개 파일 스캔 완료.")
    print(f"성공적으로 처리된 파일: {success_count}개")
    print(f"처리된 데이터 저장 위치: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="오디오 파일 길이를 15초(240,000 샘플)로 정규화합니다.")
    parser.add_argument("--input_dir", type=str, required=True, help="처리할 원본 오디오 파일이 있는 폴더")
    parser.add_argument("--output_dir", type=str, required=True, help="처리된 파일을 저장할 폴더")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)