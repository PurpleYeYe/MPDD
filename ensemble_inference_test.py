import argparse
import json
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from models.our.our_model_conformer import ourModel_conformer
from models.networks.classifier import FcClassifier
from dataset import AudioVisualDataset
from torch.utils.data import DataLoader
from utils.logger import get_logger
import torch.nn as nn

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

# ✅ 新增：使用与训练一致的 Refiner 网络结构
class EnhancedRefiner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim, nhead=3, dim_feedforward=128,
                dropout=0.1, batch_first=True
            ),
            num_layers=2
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x), None

def load_model(opt_path, model_path, device, input_dim_a, input_dim_v, labelcount, feature_max_len):
    with open(opt_path, 'r') as f:
        config = json.load(f)
    config['input_dim_a'] = input_dim_a
    config['input_dim_v'] = input_dim_v
    config['emo_output_dim'] = labelcount
    config['feature_max_len'] = feature_max_len
    config['isTrain'] = False

    opt = Opt(config)
    model = ourModel_conformer(opt)

    state_dict = torch.load(model_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        print(f"⚠️  Warning: Unexpected keys skipped in state_dict: {unexpected_keys}")

    model.to(device)
    model.eval()
    return model, opt

def get_input_dims(data_root, splitwindow, feature_pair):
    audio_type, video_type = feature_pair.split('+')
    audio_path = os.path.join(data_root, 'Testing', splitwindow, 'Audio', audio_type)
    video_path = os.path.join(data_root, 'Testing', splitwindow, 'Visual', video_type)

    for f in os.listdir(audio_path):
        if f.endswith('.npy'):
            input_dim_a = np.load(os.path.join(audio_path, f)).shape[1]
            break
    for f in os.listdir(video_path):
        if f.endswith('.npy'):
            input_dim_v = np.load(os.path.join(video_path, f)).shape[1]
            break
    return input_dim_a, input_dim_v

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def load_refiner(input_dim, output_dim, device, model_path):
    model = EnhancedRefiner(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_rootpath', type=str, required=True)
    parser.add_argument('--splitwindow_time', type=str, required=True)
    parser.add_argument('--labelcount', type=int, default=2)
    parser.add_argument('--feature_max_len', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--test_json', type=str, required=True)
    parser.add_argument('--personalized_features_file', type=str, required=True)
    parser.add_argument('--model_paths', nargs=3, required=True)
    parser.add_argument('--opt_paths', nargs=3, required=True)
    parser.add_argument('--feature_pairs', nargs=3, required=True)
    parser.add_argument('--refiner_model', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default='answer_Track2/ensemble_submission.csv')
    args = parser.parse_args()

    preds_all = []
    test_data = json.load(open(args.test_json, 'r'))

    for i in range(3):
        audio_type, video_type = args.feature_pairs[i].split('+')
        audio_path = os.path.join(args.data_rootpath, 'Testing', args.splitwindow_time, 'Audio', audio_type)
        video_path = os.path.join(args.data_rootpath, 'Testing', args.splitwindow_time, 'Visual', video_type)

        input_dim_a, input_dim_v = get_input_dims(args.data_rootpath, args.splitwindow_time, args.feature_pairs[i])

        model, opt = load_model(args.opt_paths[i], args.model_paths[i], args.device,
                                input_dim_a, input_dim_v, args.labelcount, args.feature_max_len)

        loader = DataLoader(
            AudioVisualDataset(test_data, args.labelcount, args.personalized_features_file, args.feature_max_len,
                               batch_size=args.batch_size, audio_path=audio_path, video_path=video_path,
                               isTest=True),
            batch_size=args.batch_size, shuffle=False
        )

        all_logits = []
        for batch in loader:
            model.set_input(batch)
            model.forward()
            logits = model.emo_logits.detach().cpu().numpy()
            all_logits.append(logits)
        all_logits = np.concatenate(all_logits, axis=0)
        preds_all.append(all_logits)

    # [B, C] * 3 -> [B, 3C]
    concat_logits = np.concatenate(preds_all, axis=1)

    if args.refiner_model:
        refiner = load_refiner(input_dim=concat_logits.shape[1], output_dim=args.labelcount,
                               device=args.device, model_path=args.refiner_model)
        concat_tensor = torch.tensor(concat_logits, dtype=torch.float32).to(args.device)
        refined_logits, _ = refiner(concat_tensor)
        final_preds = torch.argmax(refined_logits, dim=1).cpu().numpy()
    else:
        mean_logits = np.mean(np.stack(preds_all), axis=0)
        final_preds = np.argmax(softmax(mean_logits), axis=1)

    test_ids = ['_'.join([part.lstrip('0') for part in item["audio_feature_path"].replace(".npy", "").split('_')]) for item in test_data]

    df = pd.DataFrame({"ID": test_ids, "Ensemble_Pred": final_preds})
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"✅ Ensemble prediction saved to {args.output_csv}")

if __name__ == '__main__':
    main()
