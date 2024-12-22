# %% [markdown]
# # **第3章　学校のクラス編成**

# %% [markdown]
# ### **3.3 数理モデリングと実装**

# %% [markdown]
# ### ②データの確認

# %%
# データ処理のためのライブラリpandasの取り込み
import pandas as pd


# %% [markdown]
# (1)生徒データ(students.csv)の確認

# %%
# students.csvからの生徒データの取得
s_df = pd.read_csv("../data/students.csv")
print(len(s_df))
s_df.head()

# %%
# 学籍番号の確認
s_df["student_id"]

# %%
# 最大値の確認
s_df["student_id"].max()

# %%
# 最小値の確認
s_df["student_id"].min()

# %%
# 1〜318まで隙間なく番号が振られているかの確認
set(range(1, 319)) == set(s_df["student_id"].tolist())

# %%
# 性別（gender）の確認
s_df["gender"].value_counts()

# %%
# 学力試験の点数(score)の統計量の確認
s_df["score"].describe()

# %%
# 学力試験の点数(score)の分布の確認
s_df["score"].hist()

# %%
# リーダー気質フラグ(leader_flag)の確認
s_df["leader_flag"].value_counts()

# %%
# 特別支援フラグ(support_flag)の確認
s_df["support_flag"].value_counts()

# %% [markdown]
# (2)特定ペアデータ(student_pairs.csv)の確認

# %%
# student_pairs.csvからの特定ペアデータの取得
s_pair_df = pd.read_csv("../data/student_pairs.csv")
print(len(s_pair_df))
s_pair_df

# %% [markdown]
# ### ③数理モデリングと実装

# %%
# Pythonライブラリpython-mipの取り込み
import mip


# PythonライブラリPuLPの取り込み
# import pulp

# %%
# 数理モデルのインスタンス作成
prob = mip.Model()
# prob = pulp.LpProblem('ClassAssignmentProblem', pulp.LpMaximize)

# %%
# 生徒のリスト
S = s_df["student_id"].tolist()
print(S)

# %%
# クラスのリスト
C = ["A", "B", "C", "D", "E", "F", "G", "H"]
C

# %%
# 生徒とクラスのペアのリスト
SC = [(s, c) for s in S for c in C]
print(SC[:30])

# %%
# 生徒をどのクラスに割り当てるかを変数として定義
x = {sc: prob.add_var(var_type="B") for sc in SC}
# x = pulp.LpVariable.dicts('x', SC, cat='Binary')

# %%
# (1)各生徒は１つのクラスに割り当てる
for s in S:
    prob += mip.xsum([x[s, c] for c in C]) == 1
    # prob += pulp.lpSum([x[s,c] for c in C]) == 1

# %%
# (2)各クラスの生徒の人数は39人以上、40人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S]) >= 39
    prob += mip.xsum([x[s, c] for s in S]) <= 40
    # prob += pulp.lpSum([x[s,c] for s in S]) >= 39
    # prob += pulp.lpSum([x[s,c] for s in S]) <= 40

# %%
# 男子生徒のリスト
S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]

# 女子生徒のリスト
S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]

# (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_male]) <= 20
    prob += mip.xsum([x[s, c] for s in S_female]) <= 20
    # prob += pulp.lpSum([x[s,c] for s in S_male]) <= 20
    # prob += pulp.lpSum([x[s,c] for s in S_female]) <= 20

# %%
# 学力を辞書表現に変換
score = {row.student_id: row.score for row in s_df.itertuples()}

# 平均点の算出
score_mean = s_df["score"].mean()
print(score_mean)

# (4) 各クラスの学力試験の平均点は学年平均点±10点とする。
for c in C:
    prob += (score_mean - 10) * mip.xsum([x[s, c] for s in S]) <= mip.xsum(
        [x[s, c] * score[s] for s in S]
    )
    prob += mip.xsum([x[s, c] * score[s] for s in S]) <= (score_mean + 10) * mip.xsum(
        [x[s, c] for s in S]
    )
    # prob += (score_mean - 10) * pulp.lpSum([x[s,c] for s in S]) <= pulp.lpSum([x[s,c] * score[s] for s in S])
    # prob += pulp.lpSum([x[s,c] * score[s] for s in S]) <= (score_mean + 10) * pulp.lpSum([x[s,c] for s in S])

# %%
# リーダー気質の生徒の集合
S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]

# (5)各クラスにリーダー気質の生徒を2人以上割り当てる。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_leader]) >= 2
    # prob += pulp.lpSum([x[s,c] for s in S_leader]) >= 2

# %%
# 特別な支援が必要な生徒の集合
S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]

# (6) 特別な支援が必要な生徒は各クラスに1人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_support]) <= 1
    # prob += pulp.lpSum([x[s,c] for s in S_support]) <= 1

# %%
# 生徒の特定ペアリスト
SS = [(row.student_id1, row.student_id2) for row in s_pair_df.itertuples()]

# (7) 特定ペアの生徒は同一クラスに割り当てない。
for s1, s2 in SS:
    for c in C:
        prob += x[s1, c] + x[s2, c] <= 1

# %%
# 求解
prob.optimize()
print(prob.status)
# status = prob.solve()
# print(status)
# print(pulp.LpStatus[status])

# %%
# 最適化結果の表示
# 各クラスに割り当てられている生徒のリストを辞書に格納
C2Ss = {}
for c in C:
    C2Ss[c] = [s for s in S if x[s, c].x == 1]
    # C2Ss[c] = [s for s in S if x[s,c].value()==1]

for c, Ss in C2Ss.items():
    print("Class:", c)
    print("Num:", len(Ss))
    print("Student:", Ss)
    print()

# %%
import mip
import pandas as pd


s_df = pd.read_csv("../data/students.csv")
s_pair_df = pd.read_csv("../data/student_pairs.csv")

prob = mip.Model()

# 生徒のリスト
S = s_df["student_id"].tolist()

# クラスのリスト
C = ["A", "B", "C", "D", "E", "F", "G", "H"]

# 生徒とクラスのペアのリスト
SC = [(s, c) for s in S for c in C]

# 生徒をどのクラスに割り当てるを変数として定義
x = {sc: prob.add_var(var_type="B") for sc in SC}

# (1)各生徒は１つのクラスに割り当てる
for s in S:
    prob += mip.xsum([x[s, c] for c in C]) == 1

# (2)各クラスの生徒の人数は39人以上、40人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S]) >= 39
    prob += mip.xsum([x[s, c] for s in S]) <= 40

# 男子生徒のリスト
S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]

# 女子生徒のリスト
S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]

# (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_male]) <= 20
    prob += mip.xsum([x[s, c] for s in S_female]) <= 20

# 学力を辞書表現に変換
score = {row.student_id: row.score for row in s_df.itertuples()}

# 平均点の算出
score_mean = s_df.score.mean()

# (4) 各クラスの学力試験の平均点は学年平均点±10点とする。
for c in C:
    prob += mip.xsum([x[s, c] * score[s] for s in S]) >= (score_mean - 10) * mip.xsum(
        [x[s, c] for s in S]
    )
    prob += mip.xsum([x[s, c] * score[s] for s in S]) <= (score_mean + 10) * mip.xsum(
        [x[s, c] for s in S]
    )

# リーダー気質の生徒の集合
S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]

# (5)各クラスにリーダー気質の生徒を2人以上割り当てる。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_leader]) >= 2

# 特別な支援が必要な生徒の集合
S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]

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

#####
# import pandas as pd
# import pulp
#
# s_df = pd.read_csv('students.csv')
# s_pair_df = pd.read_csv('student_pairs.csv')
#
# prob = pulp.LpProblem('ClassAssignmentProblem', pulp.LpMaximize)
#
## 生徒のリスト
# S = s_df['student_id'].tolist()
#
## クラスのリスト
# C = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
#
## 生徒とクラスのペアのリスト
# SC = [(s,c) for s in S for c in C]
#
## 生徒をどのクラスに割り当てるを変数として定義
# x = pulp.LpVariable.dicts('x', SC, cat='Binary')
#
## (1)各生徒は１つのクラスに割り当てる
# for s in S:
#    prob += pulp.lpSum([x[s,c] for c in C]) == 1
#
## (2)各クラスの生徒の人数は39人以上、40人以下とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S]) >= 39
#    prob += pulp.lpSum([x[s,c] for s in S]) <= 40
#
## 男子生徒のリスト
# S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]
#
## 女子生徒のリスト
# S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]
#
## (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S_male]) <= 20
#    prob += pulp.lpSum([x[s,c] for s in S_female]) <= 20
#
## 学力を辞書表現に変換
# score = {row.student_id:row.score for row in s_df.itertuples()}
#
## 平均点の算出
# score_mean = s_df.score.mean()
#
## (4) 各クラスの学力試験の平均点は学年平均点±10点とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c]*score[s] for s in S]) >= (score_mean - 10) * pulp.lpSum([x[s,c] for s in S])
#    prob += pulp.lpSum([x[s,c]*score[s] for s in S]) <= (score_mean + 10) * pulp.lpSum([x[s,c] for s in S])
#
## リーダー気質の生徒の集合
# S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]
#
## (5)各クラスにリーダー気質の生徒を2人以上割り当てる。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S_leader]) >= 2
#
## 特別な支援が必要な生徒の集合
# S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]
#
## (6) 特別な支援が必要な生徒は各クラスに1人以下とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S_support]) <= 1
#
#    # 生徒の特定ペアリスト
# SS = [(row.student_id1, row.student_id2) for row in s_pair_df.itertuples()]
#
## (7) 特定ペアの生徒は同一クラスに割り当てない。
# for row in s_pair_df.itertuples():
#    s1 = row.student_id1
#    s2 = row.student_id2
#    for c in C:
#        prob += x[s1,c] + x[s2,c] <= 1
#
## 求解
# status = prob.solve()
# print('Status:', pulp.LpStatus[status])
#
## 最適化結果の表示
## 各クラスに割り当てられている生徒のリストを辞書に格納
# C2Ss = {}
# for c in C:
#    C2Ss[c] = [s for s in S if x[s,c].value()==1]
#
# for c, Ss in C2Ss.items():
#    print('Class:', c)
#    print('Num:', len(Ss))
#    print('Student:', Ss)
#    print()

# %% [markdown]
# ### 3.4 **数理モデルの検証**

# %% [markdown]
# ### ①解が要件を満たしているかどうか確認する

# %% [markdown]
# 要件(1) 学年の全生徒をそれぞれ①つのクラスに割り当てる

# %%
for s in S:
    # 割り当てられたクラスを取得
    assigned_class = [c for c in C if x[s, c].x == 1]
    # assigned_class = [c for c in C if x[s,c].value()==1]

    # 1つのクラスに割り当てられているか確認
    if len(assigned_class) != 1:
        print("error:", s, assigned_class)

# %% [markdown]
# 検証用データフレームの作成

# %%
# 検証用のデータフレームの用意
result_df = s_df.copy()

# 各生徒がどのクラスに割り当てられたかの情報を辞書に格納
S2C = {s: c for s in S for c in C if x[s, c].x == 1}
# S2C = {s:c for s in S for c in C if x[s,c].value()==1}

# 生徒データに各生徒がどのクラスに割り当てられたかの情報を結合
result_df["assigned_class"] = result_df["student_id"].map(S2C)
result_df.head(5)

# %% [markdown]
# 要件(2) 各クラスの生徒の人数は39人以上、40人以下とする

# %%
result_df.groupby("assigned_class")["student_id"].count()

# %% [markdown]
# (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。

# %%
result_df.groupby(["assigned_class", "gender"])["student_id"].count()

# %% [markdown]
# (4) 各クラスの学力試験の平均点は学年平均点±10点とする。

# %%
result_df.groupby("assigned_class")["score"].mean()

# %% [markdown]
# (5) 各クラスにリーダー気質の生徒を2人以上割り当てる。

# %%
result_df.groupby(["assigned_class"])["leader_flag"].sum()

# %% [markdown]
# (6) 特別な支援が必要な生徒は各クラスに1人以下とする。

# %%
result_df.groupby(["assigned_class"])["support_flag"].sum()

# %% [markdown]
# (7) 特定ペアの生徒は同一クラスに割り当てない。

# %%
for i, (s1, s2) in enumerate(SS):
    print("case:", i)
    c1 = S2C[s1]
    c2 = S2C[s2]
    print(f"s1:{s1}-{c1}")
    print(f"s2:{s2}-{c2}")
    print()

# %%
moment_df = result_df.groupby("assigned_class")["score"].agg([
    ('mean', 'mean'),     # 平均値
    ('std', 'std'),  # 標準偏差
    ('skewness', lambda x: x.skew()),  # 歪度
    ('kurtosis', lambda x: x.kurtosis())  # 尖度
])

# %%
moment_df.apply(lambda x: x.max() - x.min(), axis=0)

# %%

# %% [markdown]
# ### ②設定した制約の見直しと課題の洗い出し

# %%
# データの可視化ライブラリmatplotlibの取り込み
import matplotlib.pyplot as plt


# %%
fig = plt.figure(figsize=(12, 20))
for i, c in enumerate(C):
    cls_df = result_df[result_df["assigned_class"] == c]
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

# %% [markdown]
# ### ③制約の改善と数理最適化モデルの修正

# %%
# 初期クラス編成のデータを作成
# 学力をもとに順位を付与
s_df["score_rank"] = s_df["score"].rank(ascending=False, method="first")

# 学力順にクラス編成し、init_assigned_classカラムを作成
class_dic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
s_df["init_assigned_class"] = s_df["score_rank"].map(lambda x: x % 8).map(class_dic)
s_df.head()

# %%
# init_flagを作成
init_flag = {(s, c): 0 for s in S for c in C}

for row in s_df.itertuples():
    init_flag[row.student_id, row.init_assigned_class] = 1

# %%
fig = plt.figure(figsize=(12, 20))
for i, c in enumerate(C):
    cls_df = s_df[s_df["init_assigned_class"] == c]
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
# 目的関数:初期クラス編成と最適化結果のクラス編成をできるだけ一致させる
prob.objective = mip.maximize(mip.xsum([x[s, c] * init_flag[s, c] for s, c in SC]))
# prob += pulp.lpSum([x[s,c] * init_flag[s,c] for s,c in SC])

# %% [markdown]
# ### コード全体

# %%
import mip
import pandas as pd


s_df = pd.read_csv("../data/students.csv")
s_pair_df = pd.read_csv("../data/student_pairs.csv")

prob = mip.Model()

# 生徒のリスト
S = s_df["student_id"].tolist()

# クラスのリスト
C = ["A", "B", "C", "D", "E", "F", "G", "H"]

# 生徒とクラスのペアのリスト
SC = [(s, c) for s in S for c in C]

# 生徒をどのクラスに割り当てるを変数として定義
x = {sc: prob.add_var(var_type="B") for sc in SC}

# (1)各生徒は１つのクラスに割り当てる
for s in S:
    prob += mip.xsum([x[s, c] for c in C]) == 1

# (2)各クラスの生徒の人数は39人以上、40人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S]) >= 39
    prob += mip.xsum([x[s, c] for s in S]) <= 40

# 男子生徒のリスト
S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]

# 女子生徒のリスト
S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]

# (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_male]) <= 20
    prob += mip.xsum([x[s, c] for s in S_female]) <= 20

# 学力を辞書表現に変換
score = {row.student_id: row.score for row in s_df.itertuples()}

# 平均点の算出
score_mean = s_df.score.mean()

# (4) 各クラスの学力試験の平均点は学年平均点±10点とする。
for c in C:
    prob += mip.xsum([x[s, c] * score[s] for s in S]) >= (score_mean - 10) * mip.xsum(
        [x[s, c] for s in S]
    )
    prob += mip.xsum([x[s, c] * score[s] for s in S]) <= (score_mean + 10) * mip.xsum(
        [x[s, c] for s in S]
    )

# リーダー気質の生徒の集合
S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]

# (5)各クラスにリーダー気質の生徒を2人以上割り当てる。
for c in C:
    prob += mip.xsum([x[s, c] for s in S_leader]) >= 2

# 特別な支援が必要な生徒の集合
S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]

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

# 初期クラス編成を作成
s_df["score_rank"] = s_df["score"].rank(ascending=False, method="first")
class_dic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
s_df["init_assigned_class"] = s_df["score_rank"].map(lambda x: x % 8).map(class_dic)
init_flag = {(s, c): 0 for s in S for c in C}
for row in s_df.itertuples():
    init_flag[row.student_id, row.init_assigned_class] = 1

# 目的関数:初期クラス編成と最適化結果のクラス編成をできるだけ一致させる
prob.objective = mip.maximize(mip.xsum([x[s, c] * init_flag[s, c] for s, c in SC]))

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

#####
# import pandas as pd
# import pulp
#
# s_df = pd.read_csv('students.csv')
# s_pair_df = pd.read_csv('student_pairs.csv')
#
# prob = pulp.LpProblem('ClassAssignmentProblem', pulp.LpMaximize)
#
## 生徒のリスト
# S = s_df['student_id'].tolist()
#
## クラスのリスト
# C = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
#
## 生徒とクラスのペアのリスト
# SC = [(s,c) for s in S for c in C]
#
## 生徒をどのクラスに割り当てるを変数として定義
# x = pulp.LpVariable.dicts('x', SC, cat='Binary')
#
## (1)各生徒は１つのクラスに割り当てる
# for s in S:
#    prob += pulp.lpSum([x[s,c] for c in C]) == 1
#
## (2)各クラスの生徒の人数は39人以上、40人以下とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S]) >= 39
#    prob += pulp.lpSum([x[s,c] for s in S]) <= 40
#
## 男子生徒のリスト
# S_male = [row.student_id for row in s_df.itertuples() if row.gender == 1]
#
## 女子生徒のリスト
# S_female = [row.student_id for row in s_df.itertuples() if row.gender == 0]
#
## (3) 各クラスの男子生徒、女子生徒の人数は20人以下とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S_male]) <= 20
#    prob += pulp.lpSum([x[s,c] for s in S_female]) <= 20
#
## 学力を辞書表現に変換
# score = {row.student_id:row.score for row in s_df.itertuples()}
#
## 平均点の算出
# score_mean = s_df.score.mean()
#
## (4) 各クラスの学力試験の平均点は学年平均点±10点とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c]*score[s] for s in S]) >= (score_mean - 10) * pulp.lpSum([x[s,c] for s in S])
#    prob += pulp.lpSum([x[s,c]*score[s] for s in S]) <= (score_mean + 10) * pulp.lpSum([x[s,c] for s in S])
#
## リーダー気質の生徒の集合
# S_leader = [row.student_id for row in s_df.itertuples() if row.leader_flag == 1]
#
## (5)各クラスにリーダー気質の生徒を2人以上割り当てる。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S_leader]) >= 2
#
## 特別な支援が必要な生徒の集合
# S_support = [row.student_id for row in s_df.itertuples() if row.support_flag == 1]
#
## (6) 特別な支援が必要な生徒は各クラスに1人以下とする。
# for c in C:
#    prob += pulp.lpSum([x[s,c] for s in S_support]) <= 1
#
#    # 生徒の特定ペアリスト
# SS = [(row.student_id1, row.student_id2) for row in s_pair_df.itertuples()]
#
## (7) 特定ペアの生徒は同一クラスに割り当てない。
# for row in s_pair_df.itertuples():
#    s1 = row.student_id1
#    s2 = row.student_id2
#    for c in C:
#        prob += x[s1,c] + x[s2,c] <= 1
#
## 初期クラス編成を作成
# s_df['score_rank'] = s_df['score'].rank(ascending=False, method='first')
# class_dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H'}
# s_df['init_assigned_class'] = s_df['score_rank'].map(lambda x:x % 8).map(class_dic)
# init_flag = {(s,c): 0 for s in S for c in C}
# for row in s_df.itertuples():
#    init_flag[row.student_id, row.init_assigned_class] = 1
#
## 目的関数:初期クラス編成と最適化結果のクラス編成をできるだけ一致させる
# prob += pulp.lpSum([x[s,c] * init_flag[s,c] for s,c in SC])
#
## 求解
# status = prob.solve()
# print('Status:', pulp.LpStatus[status])
#
## 最適化結果の表示
## 各クラスに割り当てられている生徒のリストを辞書に格納
# C2Ss = {}
# for c in C:
#    C2Ss[c] = [s for s in S if x[s,c].value()==1]
#
# for c, Ss in C2Ss.items():
#    print('Class:', c)
#    print('Num:', len(Ss))
#    print('Student:', Ss)
#    print()

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
moment_df = result_df2.groupby("assigned_class")["score"].agg([
    ('mean', 'mean'),     # 平均値
    ('std', 'std'),  # 標準偏差
    ('skewness', lambda x: x.skew()),  # 歪度
    ('kurtosis', lambda x: x.kurtosis())  # 尖度
])

# %%
moment_df.apply(lambda x: x.max() - x.min(), axis=0)

# %%
