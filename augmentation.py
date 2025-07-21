import os
import json
import torch
import numpy as np
import random
import argparse
import logging
from datetime import datetime
from torch.utils.data import Dataset
from collections import defaultdict

class AudioVisualDataset(Dataset):
    def __init__(self, json_data, personalized_feature_file, max_len=10,
                 batch_size=32, audio_path='', video_path='', isTest=False,
                 aug_root_path=None, is_elderly=False, augment=True, logger=None,
                 aug_method='concat'):
        """
        Args:
            aug_root_path: 增强数据保存的根目录 (如 'MPDD-Young/Training1')
            is_elderly: 是否为老年组数据集
            augment: 是否进行数据增强
            logger: 日志记录器
            aug_method: 增强方法 ('concat', 'mixup', 'combined')
        """
        self.logger = logger or logging.getLogger(__name__)
        self.data = json_data
        self.max_len = max_len
        self.batch_size = batch_size
        self.isTest = isTest
        self.personalized_features = self.load_personalized_features(personalized_feature_file)
        self.audio_path = audio_path
        self.video_path = video_path
        self.aug_root_path = aug_root_path
        self.is_elderly = is_elderly
        self.augment = augment
        self.aug_method = aug_method
        self.feature_types = ['mfccs', 'opensmile', 'wav2vec']  # 所有需要处理的特征类型
        self.time_scales = ['1s', '5s']  # 所有需要处理的时间尺度

        # 按人分组所有样本（包括抑郁和正常）
        self.person_samples = defaultdict(list)  # 按人分组的样本
        for idx, entry in enumerate(self.data):
            person_id = self.get_person_id(entry['audio_feature_path'])
            self.person_samples[person_id].append(idx)

        self.logger.info(f"数据集初始化完成，总样本数: {len(self.data)}")
        self.logger.info(f"人员数量: {len(self.person_samples)}")
        self.logger.info(f"时间尺度: {self.time_scales}")
        self.logger.info(f"特征类型: {self.feature_types}")
        self.logger.info(f"增强方法: {self.aug_method}")

        # 生成增强数据
        if not isTest and augment and aug_root_path:
            self.generate_augmented_samples()

    def get_person_id(self, filepath):
        """从文件路径中提取人员ID"""
        filename = os.path.basename(filepath)
        if self.is_elderly:
            # 老年组文件名格式: 1_A_1_audio_features.npy -> 提取"1"
            return filename.split('_')[0]
        else:
            # 青年组文件名格式: 001_001.npy -> 提取"001"
            return filename.split('_')[0]

    def get_max_index(self, person_id, time_scale, feature_type):
        """获取指定人员ID、特征类型和时间尺度的最大样本索引"""
        max_index = 0

        # 检查现有数据
        for entry in self.data:
            if self.get_person_id(entry['audio_feature_path']) == person_id:
                filename = os.path.basename(entry['audio_feature_path'])

                if self.is_elderly:
                    # 老年组: 1_A_1_audio_features.npy -> 索引为1
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        try:
                            index = int(parts[2])
                            if index > max_index:
                                max_index = index
                        except ValueError:
                            self.logger.warning(f"无法解析索引: {filename}")
                else:
                    # 青年组: 001_001.npy -> 索引为001
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        try:
                            index = int(parts[1].split('.')[0])
                            if index > max_index:
                                max_index = index
                        except ValueError:
                            self.logger.warning(f"无法解析索引: {filename}")

        # 检查增强目录中已存在的文件
        if self.aug_root_path:
            aug_dir = os.path.join(self.aug_root_path, time_scale, 'Audio', feature_type)
            if os.path.exists(aug_dir):
                for fname in os.listdir(aug_dir):
                    if self.is_elderly:
                        pattern = f"{person_id}_"
                    else:
                        pattern = f"{person_id}_"

                    if fname.startswith(pattern) and fname.endswith('.npy'):
                        if self.is_elderly:
                            parts = fname.split('_')
                            if len(parts) >= 3:
                                try:
                                    index = int(parts[2])
                                    if index > max_index:
                                        max_index = index
                                except ValueError:
                                    self.logger.warning(f"无法解析索引: {fname}")
                        else:
                            try:
                                index = int(fname.split('_')[1].split('.')[0])
                                if index > max_index:
                                    max_index = index
                            except ValueError:
                                self.logger.warning(f"无法解析索引: {fname}")

        self.logger.debug(f"人员 {person_id} 在 {time_scale}/{feature_type} 的最大索引: {max_index}")
        return max_index

    def generate_augmented_samples(self):
        """根据选择的增强方法生成增强样本"""
        if not self.aug_root_path:
            return

        if self.aug_method == 'concat':
            self._generate_concat_augmentation()
        elif self.aug_method == 'mixup':
            self._generate_mixup_augmentation()
        elif self.aug_method == 'combined':
            self._generate_combined_augmentation()
        else:
            self.logger.error(f"未知的增强方法: {self.aug_method}")

    def _generate_concat_augmentation(self):
        """生成拼接增强样本 (dataset5 方法)"""
        self.logger.info("开始生成拼接增强样本（所有人）...")
        augmented_data = []

        # 按时间尺度循环 (1s 和 5s)
        for time_scale in self.time_scales:
            self.logger.info(f"\n处理时间尺度: {time_scale}")

            # 创建增强数据目录
            audio_aug_path = os.path.join(self.aug_root_path, time_scale, 'Audio')
            os.makedirs(audio_aug_path, exist_ok=True)

            # 按特征类型循环
            for feature_type in self.feature_types:
                self.logger.info(f"\n处理特征类型: {feature_type}")
                type_aug_path = os.path.join(audio_aug_path, feature_type)
                os.makedirs(type_aug_path, exist_ok=True)

                # 按每个人处理
                for person_id, sample_indices in self.person_samples.items():
                    n = len(sample_indices)
                    if n < 2:  # 至少需要2个样本才能生成增强样本
                        self.logger.warning(f"人员 {person_id} 只有 {n} 个样本，无法生成增强样本")
                        continue

                    # 获取基础索引（在生成样本之前）
                    base_index = self.get_max_index(person_id, time_scale, feature_type)

                    # 每人生成两条增强样本，使用不同的辅助样本
                    for aug_num in range(2):
                        # 随机选择一个主样本
                        main_idx = random.choice(sample_indices)
                        main_entry = self.data[main_idx]
                        main_filename = os.path.basename(main_entry['audio_feature_path'])

                        # 随机选择一个不同的辅助样本（不能与主样本相同）
                        aux_candidates = [idx for idx in sample_indices if idx != main_idx]

                        # 如果是第二条增强样本，排除第一条使用的辅助样本
                        if aug_num == 1 and 'prev_aux' in locals():
                            aux_candidates = [idx for idx in aux_candidates if idx != prev_aux]

                        # 确保有候选样本
                        if not aux_candidates:
                            self.logger.warning(f"人员 {person_id} 没有可用的辅助样本用于第 {aug_num + 1} 条增强")
                            continue

                        # 随机选择辅助样本
                        aux_idx = random.choice(aux_candidates)
                        aux_entry = self.data[aux_idx]
                        aux_filename = os.path.basename(aux_entry['audio_feature_path'])

                        # 如果是第一条增强样本，保存使用的辅助样本
                        if aug_num == 0:
                            prev_aux = aux_idx

                        # 构建完整路径
                        main_audio_path = os.path.join(self.audio_path, time_scale, 'Audio', feature_type,
                                                       main_filename)
                        if not os.path.exists(main_audio_path):
                            self.logger.warning(f"文件不存在: {main_audio_path}")
                            continue

                        aux_audio_path = os.path.join(self.audio_path, time_scale, 'Audio', feature_type, aux_filename)
                        if not os.path.exists(aux_audio_path):
                            self.logger.warning(f"文件不存在: {aux_audio_path}")
                            continue

                        # 加载音频特征
                        try:
                            audio_feat1 = np.load(main_audio_path)
                            audio_feat2 = np.load(aux_audio_path)
                        except Exception as e:
                            self.logger.error(f"加载特征文件失败: {e}")
                            continue

                        # 确定拼接长度
                        total_len = self.max_len
                        len1 = int(total_len * 0.75)  # 75%来自主样本
                        len2 = total_len - len1  # 25%来自辅助样本

                        # 处理主样本音频
                        if audio_feat1.shape[0] < len1:
                            audio_part1 = audio_feat1
                            self.logger.debug(f"主样本特征长度不足: {audio_feat1.shape[0]} < {len1}, 使用全部特征")
                        else:
                            start1 = random.randint(0, max(0, audio_feat1.shape[0] - len1))
                            audio_part1 = audio_feat1[start1:start1 + len1]
                            self.logger.debug(f"主样本特征截取: {start1}到{start1 + len1}")

                        # 处理辅助样本音频
                        if audio_feat2.shape[0] < len2:
                            audio_part2 = audio_feat2
                            self.logger.debug(f"辅助样本特征长度不足: {audio_feat2.shape[0]} < {len2}, 使用全部特征")
                        else:
                            start2 = random.randint(0, max(0, audio_feat2.shape[0] - len2))
                            audio_part2 = audio_feat2[start2:start2 + len2]
                            self.logger.debug(f"辅助样本特征截取: {start2}到{start2 + len2}")

                        # 拼接音频特征
                        augmented_audio = np.concatenate([audio_part1, audio_part2])
                        self.logger.debug(f"拼接后特征形状: {augmented_audio.shape}")

                        # 获取下一个可用的样本索引
                        new_index = base_index + 1 + aug_num  # 为每条增强样本增加索引

                        # 生成新文件名
                        if self.is_elderly:
                            # 老年组: {id}_{session}_{index}_audio_features.npy
                            parts = main_filename.split('_')
                            session_id = parts[1] if len(parts) > 1 else "A"
                            audio_filename = f"{person_id}_{session_id}_{new_index}_audio_features.npy"
                        else:
                            # 青年组: {person_id}_{index}.npy
                            audio_filename = f"{person_id}_{new_index:03d}.npy"

                        # 保存增强特征
                        audio_save_path = os.path.join(type_aug_path, audio_filename)
                        try:
                            np.save(audio_save_path, augmented_audio)
                            self.logger.debug(f"保存增强特征: {audio_save_path}")
                        except Exception as e:
                            self.logger.error(f"保存增强特征失败: {e}")
                            continue

                        # 创建新的数据项 - 只包含文件名和所有分类标签
                        new_entry = {
                            "audio_feature_path": audio_filename,
                            "video_feature_path": os.path.basename(main_entry['video_feature_path'])
                        }

                        # 复制所有分类标签
                        for key in ['bin_category', 'tri_category', 'pen_category', 'id']:
                            if key in main_entry:
                                new_entry[key] = main_entry[key]

                        augmented_data.append(new_entry)
                        self.logger.info(
                            f"为人员 {person_id} 生成 {time_scale} {feature_type} 拼接增强样本 {aug_num + 1}: {audio_filename}")

        # 将增强数据添加到主数据集
        self.data.extend(augmented_data)
        self.logger.info(f"\n拼接增强完成! 共生成 {len(augmented_data)} 个增强样本")
        self._save_augmented_data(augmented_data)

    def _generate_mixup_augmentation(self):
        """生成Mixup融合增强样本 (dataset6 方法)"""
        self.logger.info("开始生成Mixup融合增强样本（所有人）...")
        augmented_data = []
        total_augmented = 0

        # 按时间尺度循环 (1s 和 5s)
        for time_scale in self.time_scales:
            self.logger.info(f"\n处理时间尺度: {time_scale}")

            # 创建增强数据目录
            audio_aug_path = os.path.join(self.aug_root_path, time_scale, 'Audio')
            os.makedirs(audio_aug_path, exist_ok=True)

            # 按特征类型循环
            for feature_type in self.feature_types:
                self.logger.info(f"\n处理特征类型: {feature_type}")
                type_aug_path = os.path.join(audio_aug_path, feature_type)
                os.makedirs(type_aug_path, exist_ok=True)

                # 按每个人处理
                for person_id, sample_indices in self.person_samples.items():
                    n = len(sample_indices)
                    if n < 2:  # 至少需要2个样本才能生成增强样本
                        self.logger.warning(f"人员 {person_id} 只有 {n} 个样本，无法生成增强样本")
                        continue

                    # 获取基础索引（在生成样本之前）
                    base_index = self.get_max_index(person_id, time_scale, feature_type)

                    # 为每个人生成两条增强样本
                    for aug_num in range(2):
                        # 随机选择两个不同的样本
                        idx1, idx2 = random.sample(sample_indices, 2)
                        entry1 = self.data[idx1]
                        entry2 = self.data[idx2]

                        filename1 = os.path.basename(entry1['audio_feature_path'])
                        filename2 = os.path.basename(entry2['audio_feature_path'])

                        # 构建完整路径
                        audio_path1 = os.path.join(self.audio_path, time_scale, 'Audio', feature_type, filename1)
                        audio_path2 = os.path.join(self.audio_path, time_scale, 'Audio', feature_type, filename2)

                        if not os.path.exists(audio_path1):
                            self.logger.warning(f"文件不存在: {audio_path1}")
                            continue

                        if not os.path.exists(audio_path2):
                            self.logger.warning(f"文件不存在: {audio_path2}")
                            continue

                        # 加载音频特征
                        try:
                            audio_feat1 = np.load(audio_path1)
                            audio_feat2 = np.load(audio_path2)
                        except Exception as e:
                            self.logger.error(f"加载特征文件失败: {e}")
                            continue

                        # 确定最大长度
                        max_len = min(audio_feat1.shape[0], audio_feat2.shape[0], self.max_len)

                        # 随机选择起始位置
                        start1 = random.randint(0, max(0, audio_feat1.shape[0] - max_len))
                        start2 = random.randint(0, max(0, audio_feat2.shape[0] - max_len))

                        # 截取特征段
                        segment1 = audio_feat1[start1:start1 + max_len]
                        segment2 = audio_feat2[start2:start2 + max_len]

                        # 应用Mixup融合 (0.75:0.25)
                        augmented_audio = 0.75 * segment1 + 0.25 * segment2
                        self.logger.debug(f"Mixup后特征形状: {augmented_audio.shape}")

                        # 获取下一个可用的样本索引
                        new_index = base_index + 1 + aug_num

                        # 生成新文件名
                        if self.is_elderly:
                            # 老年组: {id}_{session}_{index}_audio_features.npy
                            parts = filename1.split('_')
                            session_id = parts[1] if len(parts) > 1 else "A"
                            audio_filename = f"{person_id}_{session_id}_{new_index}_audio_features.npy"
                        else:
                            # 青年组: {person_id}_{index}.npy
                            audio_filename = f"{person_id}_{new_index:03d}.npy"

                        # 保存增强特征
                        audio_save_path = os.path.join(type_aug_path, audio_filename)
                        try:
                            np.save(audio_save_path, augmented_audio)
                            self.logger.debug(f"保存增强特征: {audio_save_path}")
                        except Exception as e:
                            self.logger.error(f"保存增强特征失败: {e}")
                            continue

                        # 创建新的数据项 - 使用第一个样本的元数据
                        new_entry = {
                            "audio_feature_path": audio_filename,
                            "video_feature_path": os.path.basename(entry1['video_feature_path'])
                        }

                        # 复制所有分类标签
                        for key in ['bin_category', 'tri_category', 'pen_category', 'id']:
                            if key in entry1:
                                new_entry[key] = entry1[key]

                        augmented_data.append(new_entry)
                        self.logger.info(
                            f"为人员 {person_id} 生成 {time_scale} {feature_type} Mixup增强样本 {aug_num + 1}: {audio_filename}")
                        total_augmented += 1

        # 将增强数据添加到主数据集
        self.data.extend(augmented_data)
        self.logger.info(f"\nMixup增强完成! 共生成 {total_augmented} 个增强样本")
        self._save_augmented_data(augmented_data)

    def _generate_combined_augmentation(self):
        """生成组合增强样本 (dataset7 方法)"""
        self.logger.info("开始生成组合增强样本(为每个人生成一组拼接+Mixup)...")
        augmented_data = []
        total_augmented = 0

        # 按时间尺度循环 (1s 和 5s)
        for time_scale in self.time_scales:
            self.logger.info(f"\n处理时间尺度: {time_scale}")

            # 创建增强数据目录
            audio_aug_path = os.path.join(self.aug_root_path, time_scale, 'Audio')
            os.makedirs(audio_aug_path, exist_ok=True)

            # 按特征类型循环
            for feature_type in self.feature_types:
                self.logger.info(f"\n处理特征类型: {feature_type}")
                type_aug_path = os.path.join(audio_aug_path, feature_type)
                os.makedirs(type_aug_path, exist_ok=True)

                # 按每个人处理
                for person_id, sample_indices in self.person_samples.items():
                    n = len(sample_indices)
                    if n < 3:  # 至少需要3个样本才能生成增强样本
                        self.logger.warning(f"人员 {person_id} 只有 {n} 个样本，无法生成增强样本")
                        continue

                    # 获取基础索引（在生成样本之前）
                    base_index = self.get_max_index(person_id, time_scale, feature_type)
                    current_index = base_index + 1
                    self.logger.info(f"人员 {person_id} 当前最大索引: {base_index}, 新样本从 {current_index} 开始")

                    # 随机选择三个不同的样本
                    main_idx, aux_idx1, aux_idx2 = random.sample(sample_indices, 3)

                    main_entry = self.data[main_idx]
                    aux_entry1 = self.data[aux_idx1]
                    aux_entry2 = self.data[aux_idx2]

                    main_filename = os.path.basename(main_entry['audio_feature_path'])
                    aux_filename1 = os.path.basename(aux_entry1['audio_feature_path'])
                    aux_filename2 = os.path.basename(aux_entry2['audio_feature_path'])

                    # 构建完整路径
                    main_audio_path = os.path.join(self.audio_path, time_scale, 'Audio', feature_type,
                                                   main_filename)
                    aux_audio_path1 = os.path.join(self.audio_path, time_scale, 'Audio', feature_type,
                                                   aux_filename1)
                    aux_audio_path2 = os.path.join(self.audio_path, time_scale, 'Audio', feature_type,
                                                   aux_filename2)

                    # 检查文件是否存在
                    paths = [main_audio_path, aux_audio_path1, aux_audio_path2]
                    if not all(os.path.exists(p) for p in paths):
                        missing = [p for p in paths if not os.path.exists(p)]
                        self.logger.warning(f"以下特征文件不存在，跳过 {person_id}: {missing}")
                        continue

                    # 加载音频特征
                    try:
                        audio_feat_main = np.load(main_audio_path)
                        audio_feat_aux1 = np.load(aux_audio_path1)
                        audio_feat_aux2 = np.load(aux_audio_path2)
                    except Exception as e:
                        self.logger.error(f"加载特征文件失败: {e}")
                        continue

                    # =================================================================
                    # 方法1: 拼接增强 (Concatenation)
                    # =================================================================
                    # 确定拼接长度
                    total_len = self.max_len
                    len_main = int(total_len * 0.75)  # 75%来自主样本
                    len_aux = total_len - len_main  # 25%来自辅助样本

                    # 处理主样本音频
                    if audio_feat_main.shape[0] < len_main:
                        audio_part_main = audio_feat_main
                        self.logger.debug(f"主样本特征长度不足: {audio_feat_main.shape[0]} < {len_main}, 使用全部特征")
                    else:
                        start_main = random.randint(0, max(0, audio_feat_main.shape[0] - len_main))
                        audio_part_main = audio_feat_main[start_main:start_main + len_main]
                        self.logger.debug(f"主样本特征截取: {start_main}到{start_main + len_main}")

                    # 处理辅助样本音频
                    if audio_feat_aux1.shape[0] < len_aux:
                        audio_part_aux = audio_feat_aux1
                        self.logger.debug(f"辅助样本特征长度不足: {audio_feat_aux1.shape[0]} < {len_aux}, 使用全部特征")
                    else:
                        start_aux = random.randint(0, max(0, audio_feat_aux1.shape[0] - len_aux))
                        audio_part_aux = audio_feat_aux1[start_aux:start_aux + len_aux]
                        self.logger.debug(f"辅助样本特征截取: {start_aux}到{start_aux + len_aux}")

                    # 拼接音频特征
                    augmented_audio_concat = np.concatenate([audio_part_main, audio_part_aux])
                    self.logger.debug(f"拼接后特征形状: {augmented_audio_concat.shape}")

                    # 生成新文件名
                    if self.is_elderly:
                        # 老年组: {id}_{session}_{index}_audio_features.npy
                        parts = main_filename.split('_')
                        session_id = parts[1] if len(parts) > 1 else "A"
                        audio_filename_concat = f"{person_id}_{session_id}_{current_index}_audio_features.npy"
                    else:
                        # 青年组: {person_id}_{index}.npy
                        audio_filename_concat = f"{person_id}_{current_index:03d}.npy"

                    # 保存增强特征
                    audio_save_path_concat = os.path.join(type_aug_path, audio_filename_concat)
                    try:
                        np.save(audio_save_path_concat, augmented_audio_concat)
                        self.logger.debug(f"保存拼接增强特征: {audio_save_path_concat}")
                    except Exception as e:
                        self.logger.error(f"保存拼接增强特征失败: {e}")
                        continue

                    # =================================================================
                    # 方法2: Mixup融合 (0.75:0.25)
                    # =================================================================
                    # 确定最大长度
                    max_len = min(audio_feat_main.shape[0], audio_feat_aux2.shape[0], self.max_len)

                    # 随机选择起始位置
                    start_main = random.randint(0, max(0, audio_feat_main.shape[0] - max_len))
                    start_aux = random.randint(0, max(0, audio_feat_aux2.shape[0] - max_len))

                    # 截取特征段
                    segment_main = audio_feat_main[start_main:start_main + max_len]
                    segment_aux = audio_feat_aux2[start_aux:start_aux + max_len]

                    # 应用Mixup融合 (0.75:0.25)
                    augmented_audio_mixup = 0.75 * segment_main + 0.25 * segment_aux
                    self.logger.debug(f"Mixup后特征形状: {augmented_audio_mixup.shape}")

                    # 生成新文件名
                    current_index += 1
                    if self.is_elderly:
                        audio_filename_mixup = f"{person_id}_{session_id}_{current_index}_audio_features.npy"
                    else:
                        audio_filename_mixup = f"{person_id}_{current_index:03d}.npy"

                    # 保存增强特征
                    audio_save_path_mixup = os.path.join(type_aug_path, audio_filename_mixup)
                    try:
                        np.save(audio_save_path_mixup, augmented_audio_mixup)
                        self.logger.debug(f"保存Mixup增强特征: {audio_save_path_mixup}")
                    except Exception as e:
                        self.logger.error(f"保存Mixup增强特征失败: {e}")
                        continue

                    # 创建新的数据项 - 只包含文件名和所有分类标签
                    new_entry_concat = {
                        "audio_feature_path": audio_filename_concat,
                        "video_feature_path": os.path.basename(main_entry['video_feature_path'])
                    }
                    new_entry_mixup = {
                        "audio_feature_path": audio_filename_mixup,
                        "video_feature_path": os.path.basename(main_entry['video_feature_path'])
                    }

                    # 复制所有分类标签
                    for key in ['bin_category', 'tri_category', 'pen_category', 'id']:
                        if key in main_entry:
                            new_entry_concat[key] = main_entry[key]
                            new_entry_mixup[key] = main_entry[key]

                    augmented_data.extend([new_entry_concat, new_entry_mixup])
                    self.logger.info(
                        f"为人员 {person_id} 生成 {time_scale} {feature_type} 增强样本: "
                        f"拼接->{audio_filename_concat}, Mixup->{audio_filename_mixup}")
                    total_augmented += 2

        # 将增强数据添加到主数据集
        self.data.extend(augmented_data)
        self.logger.info(f"\n组合增强完成! 共生成 {total_augmented} 个增强样本")
        self._save_augmented_data(augmented_data)

    def _save_augmented_data(self, augmented_data):
        """保存增强数据到JSON文件"""
        if not augmented_data:
            self.logger.warning("没有生成增强数据，跳过保存")
            return

        json_save_path = os.path.join(self.aug_root_path, 'labels', 'augmented_data.json')
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        self.logger.info(f"保存增强数据信息到: {json_save_path}")

        # 如果文件已存在，则追加数据
        if os.path.exists(json_save_path):
            try:
                with open(json_save_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                combined_data = existing_data + augmented_data
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已追加增强数据到现有JSON文件")
            except Exception as e:
                self.logger.error(f"加载现有JSON失败，将覆盖: {e}")
                with open(json_save_path, 'w', encoding='utf-8') as f:
                    json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        else:
            with open(json_save_path, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"创建新的增强数据JSON文件")

    def __len__(self):
        return len(self.data)

    def fixed_windows(self, features: torch.Tensor, fixLen=4):
        timesteps, feature_dim = features.shape
        window_size = int(torch.ceil(torch.tensor(timesteps / fixLen)))
        windows = []
        for i in range(fixLen):
            start = i * window_size
            end = min(start + window_size, timesteps)
            window = features[start:end]
            if window.size(0) > 0:
                window_aggregated = torch.mean(window, dim=0)
                windows.append(window_aggregated)
            else:
                windows.append(torch.zeros(feature_dim))
        return torch.stack(windows, dim=0)

    def pad_or_truncate(self, feature, max_len):
        if feature.shape[0] < max_len:
            padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
            feature = torch.cat((feature, padding), dim=0)
        else:
            feature = feature[:max_len]
        return feature

    def load_personalized_features(self, file_path):
        try:
            data = np.load(file_path, allow_pickle=True)
            if isinstance(data, np.ndarray) and isinstance(data[0], dict):
                self.logger.info(f"成功加载个性化特征，数量: {len(data)}")
                return {entry["id"]: entry["embedding"] for entry in data}
            else:
                self.logger.error(f"个性化特征文件格式错误: {file_path}")
                raise ValueError("Unexpected data format in the .npy file")
        except Exception as e:
            self.logger.error(f"加载个性化特征失败: {e}")
            return {}

    def __getitem__(self, idx):
        entry = self.data[idx]
        filename = os.path.basename(entry['audio_feature_path'])

        # 在训练脚本中，特征类型和时间尺度由配置指定
        # 这里只返回文件名，具体路径由训练脚本构建

        # 加载视频特征（同样只使用文件名）
        video_filename = os.path.basename(entry['video_feature_path'])

        # 返回结果，具体路径由训练脚本处理
        result = {
            'audio_filename': filename,
            'video_filename': video_filename
        }

        # 添加所有分类标签
        if 'bin_category' in entry:
            result['bin_category'] = torch.tensor(entry['bin_category'], dtype=torch.long)
        if 'tri_category' in entry:
            result['tri_category'] = torch.tensor(entry['tri_category'], dtype=torch.long)
        if 'pen_category' in entry:
            result['pen_category'] = torch.tensor(entry['pen_category'], dtype=torch.long)

        # 处理个性化特征
        if 'id' in entry:
            person_id = entry['id']
            if person_id in self.personalized_features:
                result['personalized_feat'] = torch.tensor(
                    self.personalized_features[person_id], dtype=torch.float32
                )
            else:
                result['personalized_feat'] = torch.zeros(1024, dtype=torch.float32)
                self.logger.warning(f"个性化特征未找到: {person_id}")

        return result

def setup_logger(log_dir=None):
    """设置日志记录器"""
    logger = logging.getLogger('AudioAugmentation')
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # 如果提供了日志目录，则创建文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'augmentation_{timestamp}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志文件已创建: {log_file}")
    else:
        logger.info("日志仅输出到控制台")

    return logger

def main():
    """命令行执行函数，用于直接生成增强数据"""
    parser = argparse.ArgumentParser(description='生成音频增强数据')
    parser.add_argument('--json_path', type=str, required=True, help='原始JSON文件路径')
    parser.add_argument('--audio_path', type=str, required=True, help='原始音频特征路径')
    parser.add_argument('--video_path', type=str, required=True, help='原始视频特征路径')
    parser.add_argument('--aug_root', type=str, required=True, help='增强数据保存根目录')
    parser.add_argument('--personalized_file', type=str, required=True, help='个性化特征文件路径')
    parser.add_argument('--is_elderly', action='store_true', help='是否为老年组数据集')
    parser.add_argument('--max_len', type=int, default=10, help='特征最大长度')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志文件保存目录')
    parser.add_argument('--aug_method', type=str, default='concat',
                        choices=['concat', 'mixup', 'combined'],
                        help='增强方法: concat(拼接), mixup(混合), combined(组合)')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logger(args.log_dir)
    logger.info("音频增强数据生成开始")
    logger.info(f"参数: {vars(args)}")

    # 加载JSON数据
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        logger.info(f"成功加载JSON数据，条目数: {len(json_data)}")
    except Exception as e:
        logger.error(f"加载JSON数据失败: {e}")
        return

    # 创建数据集实例，自动生成增强数据
    try:
        dataset = AudioVisualDataset(
            json_data=json_data,
            personalized_feature_file=args.personalized_file,
            max_len=args.max_len,
            audio_path=args.audio_path,
            video_path=args.video_path,
            aug_root_path=args.aug_root,
            is_elderly=args.is_elderly,
            augment=True,
            logger=logger,
            aug_method=args.aug_method
        )
        logger.info(f"{args.aug_method}增强数据生成完成！")
    except Exception as e:
        logger.exception(f"生成增强数据时发生错误: {e}")

if __name__ == '__main__':
    main()