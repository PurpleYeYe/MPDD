import pandas as pd

# 读取模型输出结果
df = pd.read_csv('./answer_Track2/submission.csv')  # 或替换为你的路径

# 提取person_id（如4705_1 → 4705）
df['person_id'] = df['ID'].apply(lambda x: x.split('_')[0])

# 创建空列表用于存储处理后样本
final_results = []

# 遍历每个用户
for person_id, group in df.groupby('person_id'):
    # ========== 二分类加权投票 ==========
    bin_vote = (group['1s_bin'] * 0.5 + group['5s_bin'] * 0.5).mean()
    final_bin = int(round(bin_vote))

    # ========== 三分类加权投票 ==========
    tri_vote = (group['1s_tri'] * 0.5 + group['5s_tri'] * 0.5).mean()
    final_tri = int(round(tri_vote))

    # 将该人的所有样本标签修改为投票结果
    group['1s_bin'] = final_bin
    group['5s_bin'] = final_bin
    group['1s_tri'] = final_tri
    group['5s_tri'] = final_tri

    final_results.append(group)

# 合并所有结果，删除person_id列
final_df = pd.concat(final_results).sort_values(by='ID')
final_df = final_df.drop(columns=['person_id'])

# 保存结果
final_df.to_csv('submission.csv', index=False)
