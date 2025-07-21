import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from dataset import AudioVisualDataset
from utils.logger import get_logger
from attention_module import HybridAttention  # 👈 加在 import 区域
import torch
# 引入机器学习模型库和加载工具
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib  # 用于加载保存的模型


class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def eval_ml(model, X):
    return model.predict(X)

def preprocess_features(audio_path, video_path, personalized_feature_file, test_data, max_len, labelcount, feature_selection='none', selector_dir=''):


    test_dataset = AudioVisualDataset(
        test_data,
        labelcount,
        personalized_feature_file,
        max_len,
        audio_path=audio_path,
        video_path=video_path,
        isTest=True
    )

    audio_dim = test_dataset[0]['A_feat'].shape[1]
    video_dim = test_dataset[0]['V_feat'].shape[1]
    audio_attn = HybridAttention(input_dim=audio_dim)
    video_attn = HybridAttention(input_dim=video_dim)

    X_test = []
    for data in test_dataset:
        A_feat = data['A_feat']
        V_feat = data['V_feat']
        P_feat = data['personalized_feat']

        attn_audio = audio_attn(A_feat).detach().numpy()
        attn_video = video_attn(V_feat).detach().numpy()
        combined_feature = np.concatenate([attn_audio, attn_video, P_feat.numpy()])
        X_test.append(combined_feature)

    X_test = np.array(X_test)

    # 加载 scaler 并应用标准化
    scaler_path = os.path.join(selector_dir, 'best_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    X_test = scaler.transform(X_test)

    # 可选：PCA 或特征选择
    if feature_selection == 'selectk':
        selector_path = os.path.join(selector_dir, 'feature_selector.pkl')
        if not os.path.exists(selector_path):
            raise FileNotFoundError(f"Feature selector not found: {selector_path}")
        selector = joblib.load(selector_path)
        X_test = selector.transform(X_test)
    elif feature_selection == 'pca':
        pca_path = os.path.join(selector_dir, 'pca.pkl')
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        pca = joblib.load(pca_path)
        X_test = pca.transform(X_test)

    return X_test, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test with Machine Learning Models")
    parser.add_argument('--labelcount', type=int, default=2, help="Number of data categories (2, 3, or 5).")
    parser.add_argument('--track_option', type=str, required=True, help="Track1 or Track2")
    parser.add_argument('--feature_max_len', type=int, required=True, help="Max length of feature.")
    parser.add_argument('--data_rootpath', type=str, required=True, help="Root path to the program dataset")
    parser.add_argument('--train_model', type=str, required=True, help="Path to the trained machine learning model")

    parser.add_argument('--test_json', type=str, required=False, help="File name of the testing JSON file")
    parser.add_argument('--personalized_features_file', type=str, help="File name of the personalized features file")

    parser.add_argument('--audiofeature_method', type=str, default='wav2vec', choices=['mfccs', 'opensmile', 'wav2vec'], help="Method for extracting audio features.")
    parser.add_argument('--videofeature_method', type=str, default='openface', choices=['openface', 'resnet', 'densenet'], help="Method for extracting video features.")
    parser.add_argument('--splitwindow_time', type=str, default='1s', help="Time window for splitted features. e.g. '1s' or '5s'")
    parser.add_argument('--feature_selection', type=str, default='none',
                        choices=['none', 'selectk', 'pca'],
                        help="Feature selection or dimensionality reduction method used in training.")
    args = parser.parse_args()

    args.test_json = os.path.join(args.data_rootpath, 'Testing', 'labels', 'Testing_files.json')
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Testing', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')

    config = load_config('config.json')
    opt = Opt(config)

    # Modify individual dynamic parameters in opt according to task category
    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len

    # Splice out feature folder paths according to incoming audio and video feature types
    audio_path = os.path.join(args.data_rootpath, 'Testing', f"{args.splitwindow_time}", 'Audio', f"{args.audiofeature_method}") + '/'
    video_path = os.path.join(args.data_rootpath, 'Testing', f"{args.splitwindow_time}", 'Visual', f"{args.videofeature_method}") + '/'

    # Get feature dimensions by loading a sample feature file
    for filename in os.listdir(audio_path):
        if filename.endswith('.npy'):
            opt.input_dim_a = np.load(os.path.join(audio_path, filename)).shape[1]
            break

    for filename in os.listdir(video_path):
        if filename.endswith('.npy'):
            opt.input_dim_v = np.load(os.path.join(video_path, filename)).shape[1]
            break

    opt.name = f'{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}'
    logger_path = os.path.join(opt.log_dir, opt.name)
    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    logger = get_logger(logger_path, 'result')

    logger.info(f"splitwindow_time={args.splitwindow_time}, audiofeature_method={args.audiofeature_method}, videofeature_method={args.videofeature_method}")
    logger.info(f"labels={opt.emo_output_dim}, feature_max_len={opt.feature_max_len}")

    # 加载机器学习模型
    best_model = joblib.load(args.train_model)
    logger.info(f"Loaded model: {args.train_model}")

    test_data = json.load(open(args.test_json, 'r'))
    label = {2: "bin", 3: "tri", 5: "pen"}[args.labelcount]

    # 数据预处理
    X_test, test_dataset = preprocess_features(
        audio_path,
        video_path,
        args.personalized_features_file,
        test_data,
        opt.feature_max_len,
        args.labelcount,
        feature_selection=args.feature_selection,
        selector_dir=os.path.dirname(args.train_model)
    )

    # 加载训练时保存的 scaler
    scaler_path = os.path.join(args.train_model.rsplit('/', 1)[0], 'best_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"StandardScaler file not found at {scaler_path}")
    scaler = joblib.load(scaler_path)


    # 测试
    pred = eval_ml(best_model, X_test)

    filenames = [item["audio_feature_path"] for item in test_data if "audio_feature_path" in item]
    IDs = [path[:path.find('.')] for path in filenames]

    # 输出结果到 CSV
    pred_col_name = f"{args.splitwindow_time}_{label}"

    result_dir = f"./answer_{args.track_option}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    csv_file = f"{result_dir}/submission.csv"

    # Get the order of the IDs in the test data to ensure consistency
    if args.track_option == 'Track1':
        test_ids = [item["audio_feature_path"].split('_')[0] + '_' + item["audio_feature_path"].split('_')[2] for item in test_data]
    elif args.track_option == 'Track2':
        test_ids = ['_'.join([part.lstrip('0') for part in item["audio_feature_path"].replace(".npy", "").split('_')]) for item in test_data]

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["ID"])

    if "ID" in df.columns:
        df = df.set_index("ID")
    else:
        df = pd.DataFrame(index=test_ids)

    df.index.name = "ID"

    pred = np.array(pred)
    if len(pred) != len(test_ids):
        logger.error(f"Prediction length {len(pred)} does not match test ID length {len(test_ids)}")
        raise ValueError("Mismatch between predictions and test IDs")

    new_df = pd.DataFrame({pred_col_name: pred}, index=test_ids)
    df[pred_col_name] = new_df[pred_col_name]
    df = df.reindex(test_ids)
    df.to_csv(csv_file)

    logger.info(f"Testing complete. Results saved to: {csv_file}.")