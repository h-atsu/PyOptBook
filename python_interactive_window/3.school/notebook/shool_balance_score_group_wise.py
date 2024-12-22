# %%
import matplotlib.pyplot as plt
import mip
import pandas as pd

from consts import ROOT


# データ読み込み
s_df = pd.read_csv(ROOT / "python_interactive_window/3.school" / "data/students.csv")
s_pair_df = pd.read_csv(
    ROOT / "python_interactive_window/3.school" / "data/student_pairs.csv"
)

# 集合定義
# 生徒のリスト
S = s_df["student_id"].tolist()

# クラスのリスト
C = ["A", "B", "C", "D", "E", "F", "G", "H"]

# 生徒とクラスのペアのリスト
SC = [(s, c) for s in S for c in C]

# 男子生徒のリスト
S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]

# 女子生徒のリスト
S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]

# リーダー気質の生徒の集合
S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]

# 特別な支援が必要な生徒の集合
S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]

# 得点グループの集合
s_df["score_group"] = pd.qcut(s_df["score"], q=5, labels=False)
P = sorted(s_df["score_group"].to_list())

# 得点グループがpの生徒の集合
S_p = {}
for p in P:
    S_p[p] = s_df[s_df["score_group"] == p]["student_id"].to_list()

# 定数定義
# 学力を辞書表現に変換
score = {row.student_id: row.score for row in s_df.itertuples()}

# 平均点の算出
score_mean = s_df.score.mean()


# モデルの定義
prob = mip.Model()

# 決定変数の定義
# 生徒をどのクラスに割り当てるを変数として定義
x = {sc: prob.add_var(var_type="B") for sc in SC}

# 得点pの偏り具合を表す変数
z = {p: prob.add_var(var_type="C", lb=0) for p in P}


# 制約条件の定義
# (0)分布偏りの制約
for p in P:
    for c in C:
        prob += -z[p] <= mip.xsum([x[s, c] for s in S_p[p]]) - int(len(S_p[p]) / len(C))
        prob += mip.xsum([x[s, c] for s in S_p[p]]) - int(len(S_p[p]) / len(C)) <= z[p]

# (1)各生徒は１つのクラスに割り当てる
for s in S:
    prob += mip.xsum([x[s, c] for c in C]) == 1

# (2)各クラスの生徒の人数は39人以上、40人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S]) >= 39
    prob += mip.xsum([x[s, c] for s in S]) <= 40


# (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_male]) <= 20
    prob += mip.xsum([x[s, c] for s in S_female]) <= 20


# (4) 各クラスの学力試験の平均点は学年平均点±10点とする。
for c in C:
    prob += mip.xsum([x[s, c] * score[s] for s in S]) >= (score_mean - 10) * mip.xsum(
        [x[s, c] for s in S]
    )
    prob += mip.xsum([x[s, c] * score[s] for s in S]) <= (score_mean + 10) * mip.xsum(
        [x[s, c] for s in S]
    )


# (5)各クラスにリーダー気質の生徒を2人以上割り当てる。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_leader]) >= 2


# (6) 特別な支援が必要な生徒は各クラスに1人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_support]) <= 1

    # 生徒の特定ペアリスト
SS = [(row.student_id1, row.student_id2) for row in s_pair_df.itertuples()]

# (7) 特定ペアの生徒は同一クラスに割り当てない。
for row in s_pair_df.itertuples():
    s1 = row.student_id1
    s2 = row.student_id2
    for c in C:
        prob += x[s1, c] + x[s2, c] <= 1

# # 初期クラス編成を作成
# s_df["score_rank"] = s_df["score"].rank(ascending=False, method="first")
# class_dic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
# s_df["init_assigned_class"] = s_df["score_rank"].map(lambda x: x % 8).map(class_dic)
# init_flag = {(s, c): 0 for s in S for c in C}
# for row in s_df.itertuples():
#     init_flag[row.student_id, row.init_assigned_class] = 1

# # 目的関数:初期クラス編成と最適化結果のクラス編成をできるだけ一致させる
# prob.objective = mip.maximize(mip.xsum([x[s, c] * init_flag[s, c] for s, c in SC]))

prob.objective = mip.minimize(mip.xsum([z[p] for p in P]))

# 求解
prob.optimize()
print("Status:", prob.status)

# 最適化結果の表示
# 各クラスに割り当てられている生徒のリストを辞書に格納
C2Ss = {}
for c in C:
    C2Ss[c] = [s for s in S if x[s, c].x == 1]

for c, Ss in C2Ss.items():
    print("Class:", c)
    print("Num:", len(Ss))
    print("Student:", Ss)
    print()

# %%
# 検証用のデータフレームの用意
result_df2 = s_df.copy()

# 各生徒がどのクラスに割り当てられたかの情報を辞書に格納
S2C = {}
for s in S:
    S2C[s] = [c for c in C if x[s, c].x == 1][0]
    # S2C[s] = [c for c in C if x[s,c].value()==1][0]

# 生徒データに各生徒が度のクラスに割り当てられたかの情報を結合
result_df2["assigned_class"] = result_df2["student_id"].map(S2C)
result_df2.head(5)

# %%
fig = plt.figure(figsize=(12, 20))
for i, c in enumerate(C):
    cls_df = result_df2[result_df2["assigned_class"] == c]
    ax = fig.add_subplot(
        4,
        2,
        i + 1,
        xlabel="score",
        ylabel="num",
        xlim=(0, 500),
        ylim=(0, 20),
        title=f"Class:{c:s}",
    )
    ax.hist(cls_df["score"], bins=range(0, 500, 40))

# %%
moment_df = result_df2.groupby("assigned_class")["score"].agg(
    [
        ("mean", "mean"),  # 平均値
        ("std", "std"),  # 標準偏差
        ("skewness", lambda x: x.skew()),  # 歪度
        ("kurtosis", lambda x: x.kurtosis()),  # 尖度
    ]
)

# %%
moment_df.apply(lambda x: x.max() - x.min(), axis=0)
# %%
