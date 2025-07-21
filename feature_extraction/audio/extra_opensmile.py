import os
import numpy as np
import opensmile

input_audio_dir = r"D:\HACI\MMchallenge\Audio_split1\Audio_split_16k"  # Directory containing audio files
output_feature_dir = r"D:\HACI\MMchallenge\Audio_split1\features\opensmile"  # Directory to save .npy feature files

os.makedirs(output_feature_dir, exist_ok=True)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

for audio_file in os.listdir(input_audio_dir):
    if audio_file.endswith((".wav", ".mp3")):
        audio_path = os.path.join(input_audio_dir, audio_file)
        feature_file = os.path.splitext(audio_file)[0] + ".npy"
        output_path = os.path.join(output_feature_dir, feature_file)

        try:
            features = smile.process_file(audio_path)

            feature_array = features.to_numpy().flatten()

            print(f"Shape of features for {audio_file}: {feature_array.shape}")

            np.save(output_path, feature_array)
            print(f"Features saved for {audio_file} as {output_path}")
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")

print("Feature extraction completed.")