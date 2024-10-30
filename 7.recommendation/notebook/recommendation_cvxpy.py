# %% [markdown]
# # **第7章 商品推薦のための興味のスコアリング**

# %% [markdown]
# ### **7.3 データ分析**

# %% [markdown]
# ### ②データの概要

# %%
import pandas as pd


# access_log.csvより閲覧履歴を取り込む
log_df = pd.read_csv("../data/access_log.csv", parse_dates=["date"])
print(len(log_df))
log_df.head()

# %%
# ユーザーID(user_id)の統計値を確認する
log_df["user_id"].value_counts().describe()

# %%
# 商品ID(item_id)の統計値を確認する
log_df["item_id"].value_counts().describe()

# %%
# 閲覧日(date)の件数を確認する
log_df["date"].value_counts()

# %% [markdown]
# ### **③ データの性質**

# %%
# 日付操作をするライブラリdatetimeを取り込む
import datetime


start_date = datetime.datetime(2015, 7, 1)
end_date = datetime.datetime(2015, 7, 7)
target_date = datetime.datetime(2015, 7, 8)
print(start_date, end_date, target_date)  # 追記

# %%
# 7月1日〜7月7日までのデータを抽出する（基準日7月8日は入っていない）
x_df = log_df[(start_date <= log_df["date"]) & (log_df["date"] <= end_date)]
print(len(x_df))
x_df.head(3)

# %%
# 基準日7月8日のデータを抽出する
y_df = log_df[log_df["date"] == target_date]
print(len(y_df))
y_df.head()

# %%
# ユーザーIDと閲覧した商品IDに対して、最新度のリストを対応させる辞書を作成する（ただし、辞書の入れ子として作成）
U2I2Rcens = {}
for row in x_df.itertuples():
    # 最新度(経過日数)の算出：基準日 - 最新の閲覧日
    rcen = (target_date - row.date).days

    # 辞書に最新度を登録
    U2I2Rcens.setdefault(row.user_id, {})
    U2I2Rcens[row.user_id].setdefault(row.item_id, [])
    U2I2Rcens[row.user_id][row.item_id].append(rcen)

# %%
# ユーザーID=2497について、閲覧した商品に対する最新度のリストを確認する
U2I2Rcens[2497]

# %%
# ユーザーIDと商品IDに対応する最新度(rcen)と頻度(freq)のデータを作成する
Rows1 = []
for user_id, I2Rcens in U2I2Rcens.items():
    for item_id, Rcens in I2Rcens.items():
        freq = len(Rcens)
        rcen = min(Rcens)
        Rows1.append((user_id, item_id, rcen, freq))
UI2RF_df = pd.DataFrame(Rows1, columns=["user_id", "item_id", "rcen", "freq"])
print(len(UI2RF_df))
UI2RF_df.head()

# %%
# 閲覧フラグ(pv_flag)の作成
y_df = y_df.drop_duplicates()
print(len(y_df))
y_df["pv_flag"] = 1
y_df.head()

# %%
# 閲覧フラグ(pv_flag)の追加
UI2RFP_df = pd.merge(
    UI2RF_df,
    y_df[["user_id", "item_id", "pv_flag"]],
    how="left",
    on=["user_id", "item_id"],
)
UI2RFP_df["pv_flag"].fillna(0, inplace=True)
print(len(UI2RFP_df))
UI2RFP_df.head()

# %%
# 最新度(rcen)と頻度(freq)の取りうる範囲を確認
print(sorted(UI2RFP_df["rcen"].unique()))
print(sorted(UI2RFP_df["freq"].unique()))

# %%
# 頻度(freq)が7以下になるようにデータを抽出(本書の都合でrcenと規模感を合わせるため)
tar_df = UI2RFP_df[UI2RFP_df["freq"] <= 7]
print(len(tar_df))
tar_df.head()

# %%
# 閲覧フラグ(pv_flag)の規模感の確認
tar_df["pv_flag"].sum()

# %%
# 最新度(recency)に関する閲覧数の集計
rcen_df = pd.crosstab(index=tar_df["rcen"], columns=tar_df["pv_flag"])
rcen_df = rcen_df.rename(columns={0: "neg", 1: "pos"})
rcen_df

# %%
# 最新度(recency)に関する再閲覧確率の算出と確認
rcen_df["N"] = rcen_df["neg"] + rcen_df["pos"]
rcen_df["prob"] = rcen_df["pos"] / rcen_df["N"]
rcen_df[["prob"]].plot.bar()
rcen_df

# %%
# 頻度(frequency)に関する閲覧数の集計
freq_df = pd.crosstab(index=tar_df["freq"], columns=tar_df["pv_flag"])
freq_df = freq_df.rename(columns={0: "neg", 1: "pos"})
freq_df

# %%
# 頻度(frequency)に関する再閲覧確率の算出と確認
freq_df["N"] = freq_df["neg"] + freq_df["pos"]
freq_df["prob"] = freq_df["pos"] / freq_df["N"]
freq_df[["prob"]].plot.bar()
freq_df

# %% [markdown]
# ### **7.4 数理モデリングと実装**

# %% [markdown]
# ### ①数理モデル

# %%
# tar_dfの(再)確認
print(len(tar_df))
tar_df.head()

# %%
# 最新度(rcen)と頻度(freq)に対する総件数と再閲覧件数の算出
RF2N = {}
RF2PV = {}
for row in tar_df.itertuples():
    RF2N.setdefault((row.rcen, row.freq), 0)
    RF2PV.setdefault((row.rcen, row.freq), 0)
    RF2N[row.rcen, row.freq] += 1
    if row.pv_flag == 1:
        RF2PV[row.rcen, row.freq] += 1

# %%
# 最新度(rcen)と頻度(freq)に対する再閲覧確率の算出
RF2Prob = {}
for rf, N in RF2N.items():
    RF2Prob[rf] = RF2PV[rf] / N

# %%
# 最新度(rcen)と頻度(freq)に対して総件数、再閲覧件数、再閲覧確率を対応付けるデータの作成
Rows3 = []
for rf, N in sorted(RF2N.items()):
    pv = RF2PV[rf]
    prob = RF2Prob[rf]
    row = (rf[0], rf[1], N, pv, prob)
    Rows3.append(row)
rf_df = pd.DataFrame(Rows3, columns=["rcen", "freq", "N", "pv", "prob"])
print(len(rf_df))
rf_df.head()

# %%
# 横持ち（テーブル形式）で確認
rf_df.pivot_table(index="rcen", columns="freq", values="prob")

# %%
# データの可視化に必要なライブラリ群を取り込む
import matplotlib.pyplot as plt
import numpy as np


# %%
# 最新度(rcen)と頻度(freq)に対して再閲覧確率を対応付ける3Dグラフの作成
Freq = rf_df["freq"].unique().tolist()
Rcen = rf_df["rcen"].unique().tolist()
Z = [
    rf_df[(rf_df["freq"] == freq) & (rf_df["rcen"] == rcen)]["prob"].iloc[0]
    for freq in Freq
    for rcen in Rcen
]
Z = np.array(Z).reshape((len(Freq), len(Rcen)))
X, Y = np.meshgrid(Rcen, Freq)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d", xlabel="rcen", ylabel="freq", zlabel="prob")
ax.plot_wireframe(X, Y, Z)

# %% [markdown]
# ### **②数理最適化モデル**

# %%
import cvxpy as cp


# %% [markdown]
# (1)`rcen`と`freq`に対して再閲覧確率`pred_prob`を推定する

# %%
# 最新度(rcen)のリストと頻度(freq)のリストの作成
R = sorted(tar_df["rcen"].unique().tolist())
F = sorted(tar_df["freq"].unique().tolist())
print(R)
print(F)

# %% [markdown]
# (2)`pred_prob`は`rcen`について単調減少する

# %%
# cvxpyのr,fの2次元の変数の作成
x = cp.Variable((len(R), len(F)))

# 制約条件のリストの作成
constraints = []

# - pred_prob[r,f] + pred_prob[r+1,f] <= 0 のモデリング
for r in range(len(R) - 1):
    for f in range(len(F)):
        constraints += [x[r][f] >= x[r + 1][f]]

# %% [markdown]
# (3)`pred_prob`は`freq`について単調増加する

# %%
# pred_prob[r,f] - pred_prob[r,f+1] <= 0 のモデリング
for r in range(len(R)):
    for f in range(len(F) - 1):
        constraints += [x[r][f] <= x[r][f + 1]]

# %% [markdown]
# (4)`pred_prob`と`prob`の二乗誤差を総件数の重み付きで最小化する

# %%
# N[r,f] * (pred_prob[r,f] - prob[r,f])^2 のモデリング
objectives = []
for r in range(len(R)):
    for f in range(len(F)):
        # 0-indexedのため、r,fに+1をする
        N = RF2N[r + 1, f + 1]
        prob = RF2Prob[r + 1, f + 1]
        objectives += [N * (x[r][f] - prob) ** 2]

# %%
# 求解
prob = cp.Problem(cp.Minimize(cp.sum(objectives)), constraints)
prob.solve()

# %%
# 最新度(rcen)と頻度(freq)に対応する推定した再閲覧確率の辞書を作成
RF2PredProb = {}
for r in range(len(R)):
    for f in range(len(F)):
        # 0-indexedのため、r,fに+1をする
        RF2PredProb[r + 1, f + 1] = x.value[r][f]
rf_df["pred_prob"] = rf_df.apply(lambda y: RF2PredProb[y["rcen"], y["freq"]], axis=1)
rf_df.head()

# %% [markdown]
# ### **7.5 数理モデルの検証**

# %% [markdown]
# ### ①数理モデルの確認

# %%
# 横持ちのデータに変更
rf_df.pivot_table(index="rcen", columns="freq", values="pred_prob")

# %%
# 最新度(rcen)と頻度(freq)に対して推定した再閲覧確率を対応付ける3Dグラフの作成
Freq = rf_df.freq.unique().tolist()
Rcen = rf_df.rcen.unique().tolist()
Z = [
    rf_df[(rf_df["freq"] == freq) & (rf_df["rcen"] == rcen)]["pred_prob"].iloc[0]
    for freq in Freq
    for rcen in Rcen
]
Z = np.array(Z).reshape((len(Freq), len(Rcen)))
X, Y = np.meshgrid(Rcen, Freq)
fig = plt.figure()
ax = fig.add_subplot(
    111, projection="3d", xlabel="rcen", ylabel="freq", zlabel="pred_prob"
)
ax.plot_wireframe(X, Y, Z)

# %% [markdown]
# ### **②数理モデルの適用**

# %%
# はじめに紹介したデータを作成
Rows4 = [("item1", 1, 6), ("item2", 2, 2), ("item3", 1, 2), ("item4", 1, 1)]
sample_df = pd.DataFrame(Rows4, columns=["item_name", "rcen", "freq"])
sample_df

# %%
# 推定した再閲覧確率を結合
pd.merge(sample_df, rf_df, left_on=["rcen", "freq"], right_on=["rcen", "freq"])

# %% [markdown]
# ### **③数理モデルのブラッシュアップ**

# %%
# 最新度(rcen)に対して、再閲覧確率の下降幅が小さくなることを確認
rcen_df["prob"].diff().plot.bar()
rcen_df["prob"].diff()

# %%
# 頻度(freq)に対して、再閲覧確率の上昇幅が小さくなることを確認
freq_df["prob"].diff().plot.bar()
freq_df["prob"].diff()

# %%
# - pred_prob[r,f] + 2 * pred_prob[r+1,f] - pred_prob[r+2,f] <= 0 のモデリング
for r in range(len(R) - 2):
    for f in range(len(F)):
        constraints += [x[r][f] - x[r + 1][f] >= x[r + 1][f] - x[r + 2][f]]

# pred_prob[r,f] - 2 * pred_prob[r,f+1] + pred_prob[r,f+2] <= 0 のモデリング
for r in range(len(R)):
    for f in range(len(F) - 2):
        constraints += [x[r][f] - x[r][f + 1] >= x[r][f + 1] - x[r][f + 2]]

# %% [markdown]
# ### 全体のコード

# %%
# cvxpyのr,fの2次元の変数の作成
x = cp.Variable((len(R), len(F)))

# 制約条件のリストの作成
constraints = []

# - pred_prob[r,f] + pred_prob[r+1,f] <= 0 のモデリング
for r in range(len(R) - 1):
    for f in range(len(F)):
        constraints += [x[r][f] >= x[r + 1][f]]

# pred_prob[r,f] - pred_prob[r,f+1] <= 0 のモデリング
for r in range(len(R)):
    for f in range(len(F) - 1):
        constraints += [x[r][f] <= x[r][f + 1]]

# - pred_prob[r,f] + 2 * pred_prob[r+1,f] - pred_prob[r+2,f] <= 0 のモデリング
for r in range(len(R) - 2):
    for f in range(len(F)):
        constraints += [x[r][f] - x[r + 1][f] >= x[r + 1][f] - x[r + 2][f]]

# pred_prob[r,f] - 2 * pred_prob[r,f+1] + pred_prob[r,f+2] <= 0 のモデリング
for r in range(len(R)):
    for f in range(len(F) - 2):
        constraints += [x[r][f] - x[r][f + 1] >= x[r][f + 1] - x[r][f + 2]]

# N[r,f] * (pred_prob[r,f] - prob[r,f])^2 のモデリング
objectives = []
for r in range(len(R)):
    for f in range(len(F)):
        # 0-indexedのため、r,fに+1をする
        N = RF2N[r + 1, f + 1]
        prob = RF2Prob[r + 1, f + 1]
        objectives += [N * (x[r][f] - prob) ** 2]

# 求解
prob = cp.Problem(cp.Minimize(cp.sum(objectives)), constraints)
prob.solve()

# %%
# 最新度(rcen)と頻度(freq)に対応する推定した再閲覧確率の辞書を作成
RF2PredProb2 = {}
for r in range(len(R)):
    for f in range(len(F)):
        # 0-indexedのため、r,fに+1をする
        RF2PredProb2[r + 1, f + 1] = x.value[r][f]
rf_df["pred_prob2"] = rf_df.apply(lambda y: RF2PredProb2[y["rcen"], y["freq"]], axis=1)
rf_df.head()

# %%
# 最新度(rcen)と頻度(freq)に対して推定した再閲覧確率を対応付ける3Dグラフの作成
Freq = rf_df["freq"].unique().tolist()
Rcen = rf_df["rcen"].unique().tolist()
Z = [
    rf_df[(rf_df["freq"] == freq) & (rf_df["rcen"] == rcen)]["pred_prob2"].iloc[0]
    for freq in Freq
    for rcen in Rcen
]
Z = np.array(Z).reshape((len(Freq), len(Rcen)))
X, Y = np.meshgrid(Rcen, Freq)
fig = plt.figure()
ax = fig.add_subplot(
    111, projection="3d", xlabel="rcen", ylabel="freq", zlabel="pred_prob2"
)
ax.plot_wireframe(X, Y, Z)

# %% [markdown]
# ### **7.7 凸二次計画問題の行列表現の補足（行列表現を利用しない）**

# %%
x = cp.Variable(3)

constraints = []

constraints += [x >= 0, x <= 1]

for i in range(2):
    constraints += [x[i] <= x[i + 1]]

for i in range(1):
    constraints += [x[i + 1] - x[i] >= x[i + 2] - x[i + 1]]

w = [100, 50, 10]
p = [0.1, 0.15, 0.25]

objective = cp.sum([w[i] * (x[i] - p[i]) ** 2 for i in range(3)])

prob = cp.Problem(cp.Minimize(objective), constraints)
prob.solve()
x.value

# %%
