import pandas as pd
from npy import stationary

binary_1s_file = './answer_Track2/ensemble_submission_1s_binary.csv'
binary_5s_file = './answer_Track2/ensemble_submission_5s_binary.csv'
ternary_1s_file = './answer_Track2/ensemble_submission_1s_ternary.csv'
ternary_5s_file = './answer_Track2/ensemble_submission_5s_ternary.csv'

# 加载四个文件
df_1s_bin = pd.read_csv(binary_1s_file).rename(columns={"Ensemble_Pred": "1s_bin"})
df_5s_bin = pd.read_csv(binary_5s_file).rename(columns={"Ensemble_Pred": "5s_bin"})
df_1s_tri = pd.read_csv(ternary_1s_file).rename(columns={"Ensemble_Pred": "1s_tri"})
df_5s_tri = pd.read_csv(ternary_5s_file).rename(columns={"Ensemble_Pred": "5s_tri"})

# 合并四个 DataFrame
df = df_1s_bin.merge(df_5s_bin, on="ID") \
              .merge(df_1s_tri, on="ID") \
              .merge(df_5s_tri, on="ID")

df = df.sort_values(by='ID')

# 提取 person_id
df['person_id'] = df['ID'].apply(lambda x: x.split('_')[0])

sorted_person_ids = df['person_id'].unique()
for index, bin_value in stationary.items():
    person_to_modify = sorted_person_ids[index]
    df.loc[df['person_id'] == person_to_modify, '1s_bin'] = bin_value

# 存储最终处理结果
final_results = []

# 遍历每个 person 分组
for person_id, group in df.groupby('person_id'):
    # 二分类加权融合（1s 和 5s 各占 0.6）
    bin_vote = (group['1s_bin'] * 0.6 + group['5s_bin'] * 0.4).mean()
    final_bin = int(round(bin_vote))

    # 所有该人的样本都统一使用最终的二分类结果
    group['1s_bin'] = final_bin
    group['5s_bin'] = final_bin

    # 三分类逻辑
    if final_bin == 0:
        group['1s_tri'] = 0
        group['5s_tri'] = 0
    else:
        # 排除值为0的三分类预测结果
        tri_1s = group[group['1s_tri'] != 0]['1s_tri']
        tri_5s = group[group['5s_tri'] != 0]['5s_tri']

        # 如果有有效的三分类结果，进行加权投票
        if not tri_1s.empty or not tri_5s.empty:
            vote_1s = tri_1s.mode().iloc[0] if not tri_1s.empty else 1  # 默认选择1（轻度抑郁）
            vote_5s = tri_5s.mode().iloc[0] if not tri_5s.empty else 1  # 默认选择1（轻度抑郁）

            tri_vote = vote_1s * 0.7 + vote_5s * 0.3
            final_tri = int(round(tri_vote))
            final_tri = max(1, final_tri)  # 避免返回 0
        else:
            final_tri = 1  # 默认最保守情况为轻度抑郁

        group['1s_tri'] = final_tri
        group['5s_tri'] = final_tri

    final_results.append(group)

# 合并并排序
final_df = pd.concat(final_results)
final_df = final_df.sort_values(by='ID')
final_df = final_df.drop(columns=['person_id'])

# 保存最终输出
final_df.to_csv('./answer_Track2/submission.csv', index=False)
