import librosa
import numpy as np
import os


def extract_and_save_mfcc(input_dir, output_dir, n_mfcc=64, frame_length=2048, hop_length=512):
    """
    从指定目录中批量提取音频文件的 64 维 MFCC 特征，并保存为 .npy 文件。

    参数：
        input_dir (str): 输入目录路径，包含所有 .wav 文件。
        output_dir (str): 输出目录路径，用于保存 .npy 文件。
        n_mfcc (int): MFCC 特征维度，默认为 64。
        frame_length (int): 每帧的长度（默认 2048，对应 ~46ms）。
        hop_length (int): 帧移的长度（默认 512，对应 ~11ms）。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有 .wav 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.wav'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace('.wav', '.npy'))

            try:
                y, sr = librosa.load(input_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
                mfcc_transposed = mfcc.T  # 转置为 (n_frames, n_mfcc)

                # 保存为 .npy 文件
                np.save(output_path, mfcc_transposed)

                # 打印文件名和特征形状
                print(f"Processed {file_name}: MFCC shape {mfcc_transposed.shape}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# 输入目录和输出目录路径
input_dir = r"D:\HACI\MMchallenge\Audio_split1\Audio_split_16k"  # 替换为输入目录路径
output_dir = r"D:\HACI\MMchallenge\Audio_split1\features\mfccs"  # 替换为输出目录路径

# 批量提取并保存特征
extract_and_save_mfcc(input_dir, output_dir)