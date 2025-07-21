import argparse
import json
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from models.our.our_model_conformer import ourModel_conformer
from models.networks.classifier import FcClassifier
from dataset import AudioVisualDataset
from utils.logger import get_logger
import torch.nn as nn
import torch.nn.functional as F

class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

class EnhancedRefiner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=3, dim_feedforward=128, dropout=0.1, batch_first=True),
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
        x = x.unsqueeze(1)  # [B, 1, D]
        x = self.transformer(x).squeeze(1)  # [B, D]
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
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def get_input_dims(data_root, splitwindow, feature_pair):
    audio_type, video_type = feature_pair.split('+')
    audio_path = os.path.join(data_root, 'Training', splitwindow, 'Audio', audio_type)
    video_path = os.path.join(data_root, 'Training', splitwindow, 'Visual', video_type)

    for f in os.listdir(audio_path):
        if f.endswith('.npy'):
            input_dim_a = np.load(os.path.join(audio_path, f)).shape[1]
            break
    for f in os.listdir(video_path):
        if f.endswith('.npy'):
            input_dim_v = np.load(os.path.join(video_path, f)).shape[1]
            break
    return input_dim_a, input_dim_v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_rootpath', type=str, required=True)
    parser.add_argument('--splitwindow_time', type=str, required=True)
    parser.add_argument('--labelcount', type=int, default=2)
    parser.add_argument('--feature_max_len', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--personalized_features_file', type=str, required=True)
    parser.add_argument('--model_paths', nargs=3, required=True)
    parser.add_argument('--opt_paths', nargs=3, required=True)
    parser.add_argument('--feature_pairs', nargs=3, required=True)
    parser.add_argument('--output_model_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    logger = get_logger("logs", suffix="train_ensemble_refiner")

    train_data = json.load(open(args.train_json, 'r'))
    device = args.device
    labelcount = args.labelcount
    feature_max_len = args.feature_max_len

    all_concat_logits = []
    all_labels = []

    for i in range(3):
        audio_type, video_type = args.feature_pairs[i].split('+')
        audio_path = os.path.join(args.data_rootpath, 'Training', args.splitwindow_time, 'Audio', audio_type)
        video_path = os.path.join(args.data_rootpath, 'Training', args.splitwindow_time, 'Visual', video_type)

        input_dim_a, input_dim_v = get_input_dims(args.data_rootpath, args.splitwindow_time, args.feature_pairs[i])
        model = load_model(args.opt_paths[i], args.model_paths[i], device, input_dim_a, input_dim_v, labelcount, feature_max_len)

        loader = DataLoader(
            AudioVisualDataset(train_data, labelcount, args.personalized_features_file, feature_max_len,
                               batch_size=args.batch_size, audio_path=audio_path, video_path=video_path,
                               isTest=False),
            batch_size=args.batch_size, shuffle=False
        )

        logits_list, label_list = [], []
        for batch in loader:
            model.set_input(batch)
            model.forward()
            logits = model.emo_logits.detach().cpu().numpy()
            if np.isnan(logits).any() or np.isinf(logits).any():
                print(f"❌ Detected NaN/Inf in model {i + 1} logits! Exiting...")
                exit()
            labels = batch['emo_label'].cpu().numpy()
            logits_list.append(logits)
            label_list.append(labels)

        all_concat_logits.append(np.concatenate(logits_list, axis=0))
        if i == 0:
            all_labels = np.concatenate(label_list, axis=0)

    input_features = np.concatenate(all_concat_logits, axis=1)
    input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
    target_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)

    model = EnhancedRefiner(input_tensor.size(1), labelcount)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    logger.info("Start training enhanced ensemble refiner...")
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        output, _ = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch + 1}/500: Loss = {loss.item():.4f}")

    os.makedirs(args.output_model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_path = os.path.join(args.output_model_dir, f"ensemble_refiner_{args.splitwindow_time}_{labelcount}labels_{timestamp}.pth")
    torch.save(model.state_dict(), output_path)
    logger.info(f"✅ Saved enhanced ensemble refiner model to {output_path}")

if __name__ == '__main__':
    main()
