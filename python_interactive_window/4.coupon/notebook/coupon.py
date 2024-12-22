# %% [markdown]
# # 第4章 割引クーポンキャンペーンの効果最大化
#

# %% [markdown]
# ## 4.3 データ理解

# %% [markdown]
# ### データの確認

# %% [markdown]
# （1） 会員データ（customers.csv）の確認

# %%
import pandas as pd


cust_df = pd.read_csv("../data/customers.csv")
cust_df.shape

# %%
# ファイル上部のレコード
cust_df.head()

# %%
# データの型
cust_df.dtypes

# %%
cust_df["age_cat"].hist()

# %%
cust_df["freq_cat"].hist()

# %%
cust_pivot_df = pd.pivot_table(
    data=cust_df,
    values="customer_id",
    columns="freq_cat",
    index="age_cat",
    aggfunc="count",
)
cust_pivot_df = cust_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
cust_pivot_df

# %%
import seaborn as sns


sns.heatmap(cust_pivot_df, annot=True, fmt="d", cmap="Blues")

# %% [markdown]
# （2） 来店率データ（visit_probability.csv）の確認

# %%
prob_df = pd.read_csv("../data/visit_probability.csv")
prob_df.shape

# %%
prob_df

# %%
import matplotlib.pyplot as plt


ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
for i, ptn in enumerate(["prob_dm1", "prob_dm2", "prob_dm3"]):
    prob_pivot_df = pd.pivot_table(
        data=prob_df, values=ptn, columns="freq_cat", index="age_cat"
    )
    prob_pivot_df = prob_pivot_df.reindex(["age~19", "age20~34", "age35~49", "age50~"])
    sns.heatmap(
        prob_pivot_df, vmin=0, vmax=1, annot=True, fmt=".0%", cmap="Blues", ax=ax[i]
    )
    ax[i].set_title(f"Visit Probability of {ptn}")
plt.show()

# %% [markdown]
# ## 4.4 数理モデリングと実装

# %% [markdown]
# ### モデリング1：会員個別送付モデル

# %%
# !pip install pulp
import pulp


problem = pulp.LpProblem(name="DiscountCouponProblem1", sense=pulp.LpMaximize)

# %% [markdown]
# 要件（1） 各会員に対してどのパターンのダイレクトメールを送付するかを決定

# %%
# 会員IDのリスト
I = cust_df["customer_id"].to_list()

# %%
# ダイレクトメールのパターンのリスト
M = [1, 2, 3]

# %%
# （1）各会員に対してどのパターンのダイレクトメールを送付するかを決定
xim = {}
for i in I:
    for m in M:
        xim[i, m] = pulp.LpVariable(name=f"xim({i},{m})", cat="Binary")
# 決定変数の数
len(xim)

# %% [markdown]
# 要件（2） 各会員に対して送付するダイレクトメールはいずれか1 パターン

# %%
# （2）各会員に対して送付するダイレクトメールはいずれか1パターン
for i in I:
    problem += pulp.lpSum(xim[i, m] for m in M) == 1

# %% [markdown]
# 要件（3） クーポン付与による来客増加数を最大化

# %%
keys = ["age_cat", "freq_cat"]
cust_prob_df = pd.merge(cust_df, prob_df, on=keys)
cust_prob_df.head()

# %%
cust_prob_ver_df = cust_prob_df.rename(
    columns={"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
).melt(id_vars=["customer_id"], value_vars=[1, 2, 3], var_name="dm", value_name="prob")
cust_prob_ver_df

# %%
Pim = cust_prob_ver_df.set_index(["customer_id", "dm"])["prob"].to_dict()

# %%
Pim[1, 1]

# %%
# （3）クーポン付与による来客増加数を最大化
problem += pulp.lpSum((Pim[i, m] - Pim[i, 1]) * xim[i, m] for i in I for m in [2, 3])

# %% [markdown]
# 要件（4） 会員の予算消費期待値の合計は100 万円以下

# %%
Cm = {1: 0, 2: 1000, 3: 2000}

# %%
# （4）会員の消費する費用の期待値の合計は100万円以下
problem += (
    pulp.lpSum(Cm[m] * Pim[i, m] * xim[i, m] for i in I for m in [2, 3]) <= 1000000
)

# %%
# セグメントのリスト
S = prob_df["segment_id"].to_list()
len(S)

# %%
# 各セグメントとそのセグメントに属する会員数を対応させる辞書の作成
Ns = cust_prob_df.groupby("segment_id")["customer_id"].count().to_dict()
print(Ns)

# %%
# 会員をキーとして属するセグメントを返す辞書
Si = cust_prob_df.set_index("customer_id")["segment_id"].to_dict()

# %% [markdown]
# 要件（5） 各パターンのダイレクトメールをそれぞれのセグメントに属する会員10% 以上に送付

# %%
# （5）各パターンのダイレクトメールをそれぞれのセグメントに属する会員数の10%以上に送付
for s in S:
    for m in M:
        problem += pulp.lpSum(xim[i, m] for i in I if Si[i] == s) >= 0.1 * Ns[s]

# %%
# 時間を計測
# モデリング1は、一部の環境ではgapRel（計算の終了判定とする上界と下界のギャップのしきい値）を指定しないと停止しない
# solver = pulp.PULP_CBC_CMD(gapRel=10e-4)

import time


time_start = time.time()
status = problem.solve()
# gapRelを指定した場合はsolve関数にて上でパラメータを指定したsolverを引数にとる
# status = problem.solve(solver)
time_stop = time.time()

# %%
print(f"ステータス:{pulp.LpStatus[status]}")
print(f"目的関数値:{pulp.value(problem.objective):.4}")
print(f"計算時間:{(time_stop - time_start):.3}(秒)")

# %%
send_dm_df = pd.DataFrame(
    [[xim[i, m].value() for m in M] for i in I],
    columns=["send_dm1", "send_dm2", "send_dm3"],
)
send_dm_df.head()

# %%
cust_send_df = pd.concat(
    [cust_df[["customer_id", "age_cat", "freq_cat"]], send_dm_df], axis=1
)
cust_send_df.head()

# %%
# 各セグメントに対するそれぞれのダイレクトメールの送付率
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
for i, ptn in enumerate(["send_dm1", "send_dm2", "send_dm3"]):
    cust_send_pivot_df = pd.pivot_table(
        data=cust_send_df,
        values=ptn,
        columns="freq_cat",
        index="age_cat",
        aggfunc="mean",
    )
    cust_send_pivot_df = cust_send_pivot_df.reindex(
        ["age~19", "age20~34", "age35~49", "age50~"]
    )
    sns.heatmap(
        cust_send_pivot_df,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax[i],
    )
    ax[i].set_title(f"{ptn}_rate")
plt.show()

# %%
# 各セグメントに対するそれぞれのダイレクトメールの送付数
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
for i, ptn in enumerate(["send_dm1", "send_dm2", "send_dm3"]):
    cust_send_pivot_df = pd.pivot_table(
        data=cust_send_df,
        values=ptn,
        columns="freq_cat",
        index="age_cat",
        aggfunc="sum",
    )
    cust_send_pivot_df = cust_send_pivot_df.reindex(
        ["age~19", "age20~34", "age35~49", "age50~"]
    )
    sns.heatmap(
        cust_send_pivot_df, annot=True, fmt=".1f", cmap="Blues", vmax=800, ax=ax[i]
    )
    ax[i].set_title(f"{ptn}_num")
plt.show()

# %% [markdown]
# ### モデリング2：セグメント送付モデル

# %%
# 数理モデルのインスタンス作成
problem = pulp.LpProblem(name="DiscountCouponProblem2", sense=pulp.LpMaximize)

# %% [markdown]
# 要件（1） 各会員に対してどのパターンのダイレクトメールを送付するかを決定

# %%
# （1）各会員に対してどのパターンのダイレクトメールを送付するかを決定
xsm = {}
# ［0,1］の変数を宣言
for s in S:
    for m in M:
        xsm[s, m] = pulp.LpVariable(
            name=f"xsm({s},{m})", lowBound=0, upBound=1, cat="Continuous"
        )
len(xsm)

# %% [markdown]
# 要件（2） 各会員に対して送付するダイレクトメールはいずれか1 パターン

# %%
# （2）各会員に対して送付するダイレクトメールはいずれか1パターン
for s in S:
    problem += pulp.lpSum(xsm[s, m] for m in M) == 1

# %% [markdown]
# 要件（3） クーポン付与による来客増加数を最大化

# %%
prob_ver_df = prob_df.rename(
    columns={"prob_dm1": 1, "prob_dm2": 2, "prob_dm3": 3}
).melt(id_vars=["segment_id"], value_vars=[1, 2, 3], var_name="dm", value_name="prob")
Psm = prob_ver_df.set_index(["segment_id", "dm"])["prob"].to_dict()

# %%
# （3）クーポン付与による来客増加数を最大化
problem += pulp.lpSum(
    Ns[s] * (Psm[s, m] - Psm[s, 1]) * xsm[s, m] for s in S for m in [2, 3]
)

# %% [markdown]
# 要件（4） 会員の予算消費期待値の合計は100 万円以下

# %%
# （4）会員の予算消費期待値の合計は100万円以下
problem += (
    pulp.lpSum(Cm[m] * Ns[s] * Psm[s, m] * xsm[s, m] for s in S for m in [2, 3])
    <= 1000000
)

# %% [markdown]
# 要件（5） 各パターンのダイレクトメールをそれぞれのセグメントに属する会員数の10% 以上に送付

# %%
# （5）各パターンのダイレクトメールをそれぞれのセグメントに属する会員数の10%以上に送付
for s in S:
    for m in M:
        problem += xsm[s, m] >= 0.1

# %%
time_start = time.time()
status = problem.solve()
time_stop = time.time()
print(f"ステータス:{pulp.LpStatus[status]}")
print(f"目的関数値:{pulp.value(problem.objective):.4}")
print(f"計算時間:{(time_stop - time_start):.3}(秒)")

# %%
send_dm_df = pd.DataFrame(
    [[xsm[s, m].value() for m in M] for s in S],
    columns=["send_prob_dm1", "send_prob_dm2", "send_prob_dm3"],
)
seg_send_df = pd.concat(
    [prob_df[["segment_id", "age_cat", "freq_cat"]], send_dm_df], axis=1
)
# 各セグメントに対するそれぞれのダイレクトメールの送付率
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
for i, ptn in enumerate(["send_prob_dm1", "send_prob_dm2", "send_prob_dm3"]):
    seg_send_pivot_df = pd.pivot_table(
        data=seg_send_df,
        values=ptn,
        columns="freq_cat",
        index="age_cat",
        aggfunc="mean",
    )
    seg_send_pivot_df = seg_send_pivot_df.reindex(
        ["age~19", "age20~34", "age35~49", "age50~"]
    )
    sns.heatmap(
        seg_send_pivot_df, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, ax=ax[i]
    )
    ax[i].set_title(f"{ptn}")
plt.show()

# %%
seg_send_df["num_cust"] = seg_send_df["segment_id"].apply(lambda x: Ns[x])
seg_send_df["send_num_dm1"] = seg_send_df["send_prob_dm1"] * seg_send_df["num_cust"]
seg_send_df["send_num_dm2"] = seg_send_df["send_prob_dm2"] * seg_send_df["num_cust"]
seg_send_df["send_num_dm3"] = seg_send_df["send_prob_dm3"] * seg_send_df["num_cust"]
seg_send_df[["segment_id", "send_num_dm1", "send_num_dm2", "send_num_dm3"]].head()

# %%
# 各セグメントに対するそれぞれのダイレクトメールの送付数
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
for i, ptn in enumerate(["send_num_dm1", "send_num_dm2", "send_num_dm3"]):
    seg_send_pivot_df = pd.pivot_table(
        data=seg_send_df, values=ptn, columns="freq_cat", index="age_cat"
    )
    seg_send_pivot_df = seg_send_pivot_df.reindex(
        ["age~19", "age20~34", "age35~49", "age50~"]
    )
    sns.heatmap(
        seg_send_pivot_df,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=800,
        ax=ax[i],
    )
    ax[i].set_title(f"{ptn}")
plt.show()

# %% [markdown]
# ## 4.5 結果の評価

# %% [markdown]
# ### モデリング3：送付率下限値最大化モデル

# %%
# 数理モデルのインスタンス作成
problem = pulp.LpProblem(name="DiscountCouponProblem3", sense=pulp.LpMaximize)

# %% [markdown]
# 要件（1）各セグメントへのダイレクトメールの送付率の下限値と各会員に対してどのダイレクトメールを送付するかを決定

# %%
# （1）各セグメントへのそれぞれパターンのダイレクトメールの送付率の下限値と各会員に対してどのダイレクトメールを送付するかを決定
# 会員に対してどのダイレクトメールを送付するか
xsm = {}
# ［0,1］の変数を宣言
for s in S:
    for m in M:
        xsm[s, m] = pulp.LpVariable(
            name=f"xsm({s},{m})", lowBound=0, upBound=1, cat="Continuous"
        )
# 各セグメントへのそれぞれパターンのダイレクトメールの送付率の下限値
y = pulp.LpVariable(name="y", lowBound=0, upBound=1, cat="Continuous")

# %% [markdown]
# 要件（3） 各セグメントへのそれぞれのパターンのダイレクトメールの送付率の下限値を最大化

# %%
# （3）各セグメントへのそれぞれパターンのダイレクトメールの送付率の下限値を最大化
problem += y

# %% [markdown]
# 要件（5） 各パターンのダイレクトメールを設定した送付率の下限値以上に送付

# %%
# （5）各パターンのダイレクトメールを設定した送付率の下限値以上に送付
for s in S:
    for m in M:
        problem += xsm[s, m] >= y

# %% [markdown]
# 要件（2）、要件（4）

# %%
# （2）各会員に対して送付するダイレクトメールはいずれか1パターン
for s in S:
    problem += pulp.lpSum(xsm[s, m] for m in M) == 1
# （4）会員の予算消費期待値の合計は100万円以下
problem += (
    pulp.lpSum(Cm[m] * Ns[s] * Psm[s, m] * xsm[s, m] for s in S for m in [2, 3])
    <= 1000000
)

# %%
status = problem.solve()
max_lowerbound = pulp.value(problem.objective)
print(f"ステータス: {pulp.LpStatus[status]}, 目的関数値:{max_lowerbound :.3}")

# %%
send_dm_df = pd.DataFrame(
    [[xsm[s, m].value() for m in M] for s in S],
    columns=["send_prob_dm1", "send_prob_dm2", "send_prob_dm3"],
)
seg_send_df = pd.concat(
    [prob_df[["segment_id", "age_cat", "freq_cat"]], send_dm_df], axis=1
)
ax = {}
fig, (ax[0], ax[1], ax[2]) = plt.subplots(1, 3, figsize=(20, 3))
for i, ptn in enumerate(["send_prob_dm1", "send_prob_dm2", "send_prob_dm3"]):
    seg_send_pivot_df = pd.pivot_table(
        data=seg_send_df,
        values=ptn,
        columns="freq_cat",
        index="age_cat",
        aggfunc="mean",
    )
    seg_send_pivot_df = seg_send_pivot_df.reindex(
        ["age~19", "age20~34", "age35~49", "age50~"]
    )
    sns.heatmap(
        seg_send_pivot_df, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, ax=ax[i]
    )
    ax[i].set_title(f"{ptn}")
plt.show()

# %%
# 数理モデルのインスタンス作成
problem = pulp.LpProblem(name="DiscountCouponProblem3", sense=pulp.LpMaximize)
# （1）各会員に対してどのダイレクトメールを送付するかを決定
xsm = {}
# ［0,1］の変数を宣言
for s in S:
    for m in M:
        xsm[s, m] = pulp.LpVariable(
            name=f"xsm({s},{m})", lowBound=0, upBound=1, cat="Continuous"
        )
# （2）各会員に対して送付するダイレクトメールはいずれか1パターン
for s in S:
    problem += pulp.lpSum(xsm[s, m] for m in M) == 1
# （3）クーポン付与による来客増加数を最大化
problem += pulp.lpSum(
    Ns[s] * (Psm[s, m] - Psm[s, 1]) * xsm[s, m] for s in S for m in [2, 3]
)
# （4）会員の予算消費期待値の合計は100万円以下
problem += (
    pulp.lpSum(Cm[m] * Ns[s] * Psm[s, m] * xsm[s, m] for s in S for m in [2, 3])
    <= 1000000
)
# （5）各パターンのダイレクトメールを設定した送付率の下限値以上に送付
for s in S:
    for m in M:
        problem += xsm[s, m] >= max_lowerbound
status = problem.solve()
print(
    f"ステータス: {pulp.LpStatus[status]}, 目的関数値:{pulp.value(problem.objective):.4}"
)

# %% [markdown]
# ### 投資対効果の評価

# %%
cost_list = []
cpa_list = []
inc_action_list = []

# %%
print("ステータス, キャンペーン費用, 来客増加数, CPA")
for cost in range(761850, 3000000, 100000):
    # 数理モデルのインスタンス作成
    problem = pulp.LpProblem(name="DiscountCouponProblem2", sense=pulp.LpMaximize)
    # （1）各会員に対してどのダイレクトメールを送付するかを決定
    xsm = {}
    # [0,1]の変数を宣言
    for s in S:
        for m in M:
            xsm[s, m] = pulp.LpVariable(
                name=f"xsm({s},{m})", lowBound=0, upBound=1, cat="Continuous"
            )
    # （2）各会員に対して送付するダイレクトメールはいずれか1パターン
    for s in S:
        problem += pulp.lpSum(xsm[s, m] for m in M) == 1
    # （3）クーポン付与による来客増加数を最大化
    problem += pulp.lpSum(
        Ns[s] * (Psm[s, m] - Psm[s, 1]) * xsm[s, m] for s in S for m in [2, 3]
    )
    # （4）会員の予算消費期待値の合計はcost円以下
    problem += (
        pulp.lpSum(Cm[m] * Ns[s] * Psm[s, m] * xsm[s, m] for s in S for m in [2, 3])
        <= cost
    )
    # （5）各パターンのダイレクトメールを設定した送付率の下限値以上に送付
    for s in S:
        for m in M:
            problem += xsm[s, m] >= 0.1
    status = problem.solve()
    cpa = cost / pulp.value(problem.objective)
    inc_action = pulp.value(problem.objective)
    cost_list.append(cost)
    cpa_list.append(cpa)
    inc_action_list.append(inc_action)
    print(f"{pulp.LpStatus[status]}, {cost}, {inc_action :.4}, {cpa :.5}")

# %%
from matplotlib import ticker


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.scatter(cost_list, inc_action_list, marker="x", label="Incremental visitor")
ax2.scatter(cost_list, cpa_list, label="CPA")
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x :,.0f}"))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x :,.0f}"))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x :,.0f}"))
ax1.set_xlabel("Cost")
ax1.set_ylabel("Incremental visitor")
ax2.set_ylabel("CPA")
reg1, label1 = ax1.get_legend_handles_labels()
reg2, label2 = ax2.get_legend_handles_labels()
ax2.legend(reg1 + reg2, label1 + label2, loc="upper center")
plt.show()

# %%
