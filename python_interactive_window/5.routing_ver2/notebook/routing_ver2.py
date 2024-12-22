# %% [markdown]
# # 5章 1つの車両を用いた多機関の配送計画

# %%
# 第二版のソースコードを以下のフォルダに配置
# PyOptBook
#   |- 5.routing_ver2(第二版のソースコード)
#       |-small_locations.cv
#       |-small_distances.csv
#       |-small_orders.csv
#       |-locations.csv
#       |-distances.csv
#       |-orders.csv

# 初版のソースコードは以下のフォルダに配置
# PyOptBook
#   |- 5.routing

# %% [markdown]
# ## 5.4. 素朴な数理モデルの実装と実験

# %% [markdown]
# ### ②データ確認

# %%
import pandas as pd


k_df = pd.read_csv("../data/small_locations.csv")
print(len(k_df))
k_df.head()

# %%
import matplotlib.pyplot as plt


fig = plt.subplot()
fig.set_aspect("equal")
for row in k_df.itertuples():
    if row.depo_flag:
        fig.scatter(row.x, row.y, marker="o")
    else:
        fig.scatter(row.x, row.y, marker="x")
    fig.annotate(row.k, (row.x, row.y + 0.1))
plt.show()

# %%
kk_df = pd.read_csv("../data/small_distances.csv")
print(len(kk_df))
kk_df.head()

# %%
r_df = pd.read_csv("../data/small_orders.csv")
print(len(r_df))
r_df.head()

# %% [markdown]
# ### ③素朴な数理モデルの実装と確認（小規模データ）

# %%
# 配送日のリスト
D = list(range(r_df.b.min(), r_df.e.max() + 1))

# 配送センター
p = k_df[k_df.depo_flag == 1].k.tolist()[0]

# お店のリスト
S = k_df[k_df.depo_flag == 0].k.tolist()

# 地点のリスト
K = [p] + S

# 荷物のリスト
R = r_df.r.tolist()

# 各変数を定義するために必要なリスト
DKK = [(d, k1, k2) for d in D for k1 in K for k2 in K]
DR = [(d, r) for d in D for r in R]
DK = [(d, k) for d in D for k in K]

# 表示
print(D)
print(K)
print(R)
print(DKK[:5])
print(DR[:5])
print(DK[:5])

# %%
# 荷物に紐づける配送先のお店
R2S = {row.r: row.s for row in r_df.itertuples()}

# 荷物に紐づける重量
R2W = {row.r: row.w for row in r_df.itertuples()}

# 荷物に紐づける指定配送期間の開始日
R2B = {row.r: row.b for row in r_df.itertuples()}

# 荷物に紐づける指定配送期間の終了日
R2E = {row.r: row.e for row in r_df.itertuples()}

# 地点間の移動時間
KK2T = {(row.k1, row.k2): row.t for row in kk_df.itertuples()}

print("R2S:", R2S)
print("R2W:", R2W)
print("R2B:", R2B)
print("R2E:", R2E)
print("KK2T:", KK2T)

# %% [markdown]
# ### ☆「素朴な数理モデル」の実装

# %%
import pulp


prob = pulp.LpProblem("vrp_small", pulp.LpMinimize)

x = pulp.LpVariable.dicts("x", DKK, cat="Binary")
u = pulp.LpVariable.dicts("u", DK, cat="Integer", lowBound=0)
y = pulp.LpVariable.dicts("y", DR, cat="Binary")
h = pulp.LpVariable.dicts("h", D, cat="Continuous", lowBound=0)

for d in D:
    for k1 in K:
        # (A-1) 各配送日について、ある地点に移動する数とその地点から移動する数は一致
        prob += pulp.lpSum([x[d, k1, k2] for k2 in K]) == pulp.lpSum(
            [x[d, k2, k1] for k2 in K]
        )

        # (A-2) 各配送日について、地点に訪問する数は高々1回まで
        prob += pulp.lpSum([x[d, k2, k1] for k2 in K]) <= 1

for d in D:
    # (B-1) 各配送日について、配送センターは出発地点(0番目に訪問)
    prob += u[d, p] == 0

    # (B-2) 各配送日における、地点への訪問順番の上下限値
    for s in S:
        prob += u[d, s] >= 1
        prob += u[d, s] <= len(K) - 1

    # (B-3) 各配送日について、お店間だけのサイクルを禁止
    for s1 in S:
        for s2 in S:
            prob += u[d, s1] + 1 <= u[d, s2] + (len(K) - 1) * (1 - x[d, s1, s2])

# (C) 各荷物は、自社配送するなら期間内で高々1回まで
for r in R:
    prob += pulp.lpSum([y[d, r] for d in D]) <= 1

# (D) 各配送日について、荷物を自社配送するなら、配送先のお店に訪問
for d in D:
    for r in R:
        tar_s = R2S[r]
        prob += y[d, r] <= pulp.lpSum(x[d, k, tar_s] for k in K)

# (E) 各配送日について、荷物の重量は4,000[kg]以下
for d in D:
    prob += pulp.lpSum([y[d, r] * R2W[r] for r in R]) <= 4000

# (F) 各配送日について、ドライバーの残業時間は所定労働時間の8時間を差し引いた労働時間
for d in D:
    prob += (
        pulp.lpSum([KK2T[k1, k2] * x[d, k1, k2] for k1 in K for k2 in K]) - 8 <= h[d]
    )

# (G) 各配送日について、ドライバーの残業時間は3時間以内
for d in D:
    prob += h[d] <= 3

# (H) 各荷物は指定配送期間外の配送を禁止
for r in R:
    for d in D:
        if d < R2B[r]:
            prob += y[d, r] == 0
        if R2E[r] < d:
            prob += y[d, r] == 0

# (I) 配送費用（残業費用+外注費用）を最小化する
zangyo = pulp.lpSum([3000 * h[d] for d in D])
gaityu = pulp.lpSum([46 * R2W[r] * (1 - pulp.lpSum(y[d, r] for d in D)) for r in R])
prob += zangyo + gaityu

# 求解（明示的にCBCソルバーを指定する）
solver = pulp.PULP_CBC_CMD()
status = prob.solve(solver)
print("ステータス:", pulp.LpStatus[status])

# 結果の表示
print(f"残業費用:{zangyo.value():.0f}[円]")
print(f"外注費用:{gaityu.value():.0f}[円]")
print(f"費用合計:{zangyo.value() + gaityu.value():.0f}[円]")

# %%
for d in D:
    X = [(k1, k2) for k1 in K for k2 in K if x[d, k1, k2].value() == 1]

    time = sum([KK2T[k1, k2] for k1, k2 in X])
    print(f"---配送日:{d}日目---")
    print(f"配送重量:{sum([y[d,r].value() * R2W[r] for r in R])}[kg]")
    print(f"移動時間:{time:.2f}[h]")
    print(f"残業時間:{h[d].value():.2f}[h]")
    print("x:", X)

    # 移動する地点の順番のリストを作成
    tar = p
    Route = [p]
    while len(X) >= 1:
        for k1, k2 in X:
            if k1 == tar:
                tar = k2
                Route.append(k2)
                X.remove((k1, k2))

    print("u:", [u[d, k].value() for k in Route])
    print("配送ルート:", "->".join(Route))

# %%
for r in R:
    # 自社トラックで配送したかどうかのフラグ
    owned_truck_flag = sum([y[d, r].value() for d in D])
    if owned_truck_flag:
        # 配送日の取得
        tar_d = [d for d in D if y[d, r].value() == 1][0]
        text = f"荷物{r}(お店{R2S[r]},{R2W[r]}[kg])-配送日:{tar_d}日目"
    else:
        # 外注費用の取得
        gc = 46 * R2W[r]
        text = f"荷物{r}(お店{R2S[r]},{R2W[r]}[kg])-外注費用:{gc}[円]"
    print(text)

# %% [markdown]
# ### ④素朴な数理モデルの実験（中規模データ）

# %%
k_df = pd.read_csv("../data/locations.csv")
print(len(k_df))
k_df.head()

# %%
fig = plt.subplot()
fig.set_aspect("equal")
for row in k_df.itertuples():
    if row.depo_flag:
        fig.scatter(row.x, row.y, marker="o")
    else:
        fig.scatter(row.x, row.y, marker="x")
    fig.annotate(row.k, (row.x, row.y + 0.1))
plt.show()

# %%
kk_df = pd.read_csv("../data/distances.csv")
print(len(kk_df))
kk_df.head()

# %%
r_df = pd.read_csv("../data/orders.csv")
print(len(r_df))
r_df.head()

# %%
# 配送日のリスト
D = list(range(r_df.b.min(), r_df.e.max() + 1))

# 配送センター
p = k_df[k_df.depo_flag == 1].k.tolist()[0]

# お店のリスト
S = k_df[k_df.depo_flag == 0].k.tolist()

# 地点のリスト
K = [p] + S

# 荷物のリスト
R = r_df.r.tolist()

# 各種変数を定義するために必要なリスト
DKK = [(d, k1, k2) for d in D for k1 in K for k2 in K]
DR = [(d, r) for d in D for r in R]
DK = [(d, k) for d in D for k in K]

# 表示
print(D)
print(K)
print(R[:5])
print(DKK[:5])
print(DR[:5])
print(DK[:5])

# %%
# 荷物に紐づける配送先のお店
R2S = {row.r: row.s for row in r_df.itertuples()}

# 荷物に紐づける重量
R2W = {row.r: row.w for row in r_df.itertuples()}

# 荷物に紐づける指定配送期間の開始日
R2B = {row.r: row.b for row in r_df.itertuples()}

# 荷物に紐づける指定配送期間の終了日
R2E = {row.r: row.e for row in r_df.itertuples()}

# 地点の移動時間
KK2T = {(row.k1, row.k2): row.t for row in kk_df.itertuples()}

# 表示
print("R2S:", R2S)
print("R2W:", R2W)
print("R2B:", R2B)
print("R2E:", R2E)
print("KK2T:", KK2T)

# %%
import pulp


prob = pulp.LpProblem("vrp", pulp.LpMinimize)

x = pulp.LpVariable.dicts("x", DKK, cat="Binary")
u = pulp.LpVariable.dicts("u", DK, cat="Integer", lowBound=0)
y = pulp.LpVariable.dicts("y", DR, cat="Binary")
h = pulp.LpVariable.dicts("h", D, cat="Continuous", lowBound=0)

for d in D:
    for k1 in K:
        # (A-1) 各配送日について、ある地点に移動する数とその地点から移動する数は一致
        prob += pulp.lpSum([x[d, k1, k2] for k2 in K]) == pulp.lpSum(
            [x[d, k2, k1] for k2 in K]
        )

        # (A-2) 各配送日について、地点に訪問する数は高々1回まで
        prob += pulp.lpSum([x[d, k2, k1] for k2 in K]) <= 1

for d in D:
    # (B-1) 各配送日について、配送センターは出発地点(0番目に訪問)
    prob += u[d, p] == 0

    # (B-2) 各配送日における、地点への訪問順番の上下限値
    for s in S:
        prob += u[d, s] >= 1
        prob += u[d, s] <= len(K) - 1

    # (B-3) 各配送日について、お店間だけのサイクルを禁止
    for s1 in S:
        for s2 in S:
            prob += u[d, s1] + 1 <= u[d, s2] + (len(K) - 1) * (1 - x[d, s1, s2])

# (C) 各荷物は、自社配送するなら期間内で高々1回まで
for r in R:
    prob += pulp.lpSum([y[d, r] for d in D]) <= 1

# (D) 各配送日について、荷物を自社配送するなら、配送先のお店に訪問
for d in D:
    for r in R:
        tar_s = R2S[r]
        prob += y[d, r] <= pulp.lpSum(x[d, k, tar_s] for k in K)

# (E) 各配送日について、荷物の重量は4,000[kg]以下
for d in D:
    prob += pulp.lpSum([y[d, r] * R2W[r] for r in R]) <= 4000

# (F) 各配送日について、ドライバーの残業時間は所定労働時間の8時間を差し引いた労働時間
for d in D:
    prob += (
        pulp.lpSum([KK2T[k1, k2] * x[d, k1, k2] for k1 in K for k2 in K]) - 8 <= h[d]
    )

# (G) 各配送日について、ドライバーの残業時間は3時間以内
for d in D:
    prob += h[d] <= 3

# (H) 各荷物は指定配送期間外の配送を禁止
for r in R:
    for d in D:
        if d < R2B[r]:
            prob += y[d, r] == 0
        if R2E[r] < d:
            prob += y[d, r] == 0

# (I) 配送費用（残業費用+外注費用）を最小化する
zangyo = pulp.lpSum([3000 * h[d] for d in D])
gaityu = pulp.lpSum([46 * R2W[r] * (1 - pulp.lpSum(y[d, r] for d in D)) for r in R])
prob += zangyo + gaityu

# 求解（明示的にCBCソルバーを指定する）
solver = pulp.PULP_CBC_CMD(gapRel=0.1)  # 終了条件として、引数 gapRel=0.1 を指定
status = prob.solve(solver)
print("ステータス:", pulp.LpStatus[status])

# 結果の表示
print(f"残業費用:{zangyo.value():.0f}[円]")
print(f"外注費用:{gaityu.value():.0f}[円]")
print(f"費用合計:{zangyo.value() + gaityu.value():.0f}[円]")

# %% [markdown]
# ### **===以下、実行環境によって出力が異なることに注意（終了条件を指定しているため）===**

# %%
for d in D:
    X = [(k1, k2) for k1 in K for k2 in K if x[d, k1, k2].value() == 1]

    time = sum([KK2T[k1, k2] for k1, k2 in X])
    print(f"---配送日:{d}日目---")
    print(f"配送重量:{sum([y[d,r].value() * R2W[r] for r in R])}[kg]")
    print(f"移動時間:{time:.2f}[h]")
    print(f"残業時間:{h[d].value():.2f}[h]")

    # 移動する地点の順番のリストを作成
    tar = p
    Route = [p]
    while len(X) >= 1:
        for k1, k2 in X:
            if k1 == tar:
                tar = k2
                Route.append(k2)
                X.remove((k1, k2))

    print("配送ルート:", "->".join(Route))

# %%
for r in R:
    # 自社のトラックで配送したかどうかのフラグ
    owned_truck_flag = sum([y[d, r].value() for d in D])
    if owned_truck_flag:
        # 配送日の取得
        tar_d = [d for d in D if y[d, r].value() == 1][0]
        text = f"荷物{r}(お店{R2S[r]},{R2W[r]}[kg])-配送日:{tar_d}日目"
    else:
        # 外注費用の取得
        gc = 46 * R2W[r]
        text = f"荷物{r}(お店{R2S[r]},{R2W[r]}[kg])-外注費用:{gc}[円]"
    print(text)

# %%
# 各配送日に対して、配送したお店を紐づける辞書の作成
resD2S = {d: set() for d in D}
for d, r in y:
    if y[d, r].value() == 1:
        resD2S[d].add(R2S[r])

for d in D:
    print(f"---配送日:{d}日目---")
    X = [(k1, k2) for k1 in K for k2 in K if x[d, k1, k2].value() == 1]
    time = sum([KK2T[k1, k2] for k1, k2 in X])
    print(f"移動時間:{time:.2f}[h]")
    routeS = set([k1 for k1, k2 in X]) - set([p])
    print("(A)配送ルートのお店:", routeS)
    targetS = resD2S[d]
    print("(B)荷物の注文があるお店:", targetS)

    print("(A)-(B):", routeS - targetS)

# %%
import matplotlib.pyplot as plt


# 各地点の座標の取得
K2XY = {row.k: (row.x, row.y) for row in k_df.itertuples()}

fig = plt.figure(figsize=(12, 20))
for i in range(len(D)):
    d = D[i]
    X = [(k1, k2) for k1 in K for k2 in K if x[d, k1, k2].value() == 1]
    routeK = [k1 for k1, k2 in X]
    time = sum(KK2T[k1, k2] for k1, k2 in X)
    title_text = f"day:{d}({time:.1f}[h])"
    ax = fig.add_subplot(
        5, 4, i + 1, xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), title=title_text
    )
    ax.set_aspect("equal")

    for row in k_df.itertuples():
        if row.k in routeK:
            if row.depo_flag:
                ax.scatter(row.x, row.y, marker="o")
            else:
                ax.scatter(row.x, row.y, marker="x")

    for k1, k2 in X:
        (x1, y1) = K2XY[k1]
        (x2, y2) = K2XY[k2]
        ax.arrow(
            x1, y1, (x2 - x1), (y2 - y1), head_width=0.2, length_includes_head=True
        )

# %% [markdown]
# ## 5.5. 配送パターンを利用した数理モデリング

# %% [markdown]
# ### ①効率的な配送パターンの列挙

# %% [markdown]
# ### （1）ステップ１　実行可能な配送ルートの列挙

# %%
import pandas as pd


k_df = pd.read_csv("../data/locations.csv")
kk_df = pd.read_csv("../data/distances.csv")
r_df = pd.read_csv("../data/orders.csv")

D = list(range(r_df.b.min(), r_df.e.max() + 1))
p = k_df[k_df.depo_flag == 1].k.tolist()[0]
S = k_df[k_df.depo_flag == 0].k.tolist()
K = [p] + S
R = r_df.r.tolist()

R2S = {row.r: row.s for row in r_df.itertuples()}
R2W = {row.r: row.w for row in r_df.itertuples()}
R2B = {row.r: row.b for row in r_df.itertuples()}
R2E = {row.r: row.e for row in r_df.itertuples()}
KK2T = {(row.k1, row.k2): row.t for row in kk_df.itertuples()}

# %%
import pulp


def tsp(p_, S_, KK2T_):
    K = [p_] + S_
    KK = [(k1, k2) for k1 in K for k2 in K]
    prob = pulp.LpProblem("tsp", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", KK, cat="Binary")
    u = pulp.LpVariable.dicts("u", K, cat="Integer", lowBound=0)

    # 各地点に必ず1回訪問する
    for k1 in K:
        prob += pulp.lpSum([x[k1, k2] for k2 in K]) == 1
        prob += pulp.lpSum([x[k2, k1] for k2 in K]) == 1

    # (B-1)
    prob += u[p_] == 0

    # (B-2)
    for s in S_:
        prob += u[s] >= 1
        prob += u[s] <= len(K) - 1

    # (B-3)
    for s1 in S_:
        for s2 in S_:
            prob += u[s1] + 1 <= u[s2] + (len(K) - 1) * (1 - x[s1, s2])

    # 移動時間は11時間以内
    prob += pulp.lpSum([KK2T_[k1, k2] * x[k1, k2] for k1 in K for k2 in K]) <= 11

    # 目的関数は移動時間の最小化
    prob += pulp.lpSum([KK2T_[k1, k2] * x[k1, k2] for k1 in K for k2 in K])

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if status == 1:
        X = [(k1, k2) for k1 in K for k2 in K if x[k1, k2].value() == 1]
        time = sum([KK2T_[k1, k2] for k1, k2 in X])
    else:
        X = []
        time = -1
    return status, X, time


# %%
testS = ["s4", "s5", "s9"]
status, X, time = tsp(p, testS, KK2T)
print("ステータス:", pulp.LpStatus[status])
print("x:", X)
print("移動時間:", time)

# %%
testS = ["s1", "s2", "s3", "s4", "s5", "s9"]
status, X, time = tsp(p, testS, KK2T)
print("ステータス:", pulp.LpStatus[status])
print("x:", X)
print("移動時間:", time)

# %%
from itertools import combinations


Stp2Pat = {}
for n in range(len(S) + 1):
    # お店の集合Sからn個選ぶ組合せを列挙
    count = 0
    for Stp in combinations(S, n):  # 辞書式で列挙される
        # 配送センターpとn個のお店(Stp)の巡回セールスマン問題を解く
        status, X, time = tsp(p, list(Stp), KK2T)

        # 解が存在するときのみ配送ルートとして採用
        if status == 1:
            Stp2Pat[Stp] = (X, time)
            count += 1
    print(f"訪問するお店の数:{n} 配送ルート数:{count}")
print("要件(ⅰ)(ⅱ)を満たす配送ルート数:", len(Stp2Pat))

# %%
for i, (Stp, Pat) in enumerate(Stp2Pat.items()):
    if i > 3:
        break
    print("---お店の組合せ:", Stp)
    print("配送ルート:", Pat)

# %% [markdown]
# ### （2）ステップ２　効率的な配送パターンの列挙

# %%
# 配送日に対して配送可能な荷物のリストを紐づける辞書
D2R = {d: [] for d in D}
for r in R:
    for d in range(R2B[r], R2E[r] + 1):
        D2R[d].append(r)

# 各配送日に配送可能な荷物のリスト
for d in D:
    print("配送日:", d, D2R[d])

# %%
# 各配送日に紐づける効率的な配送パターンのリスト
D2Pat = {d: [] for d in D}
for d in D:
    # 配送日dに配送可能な荷物のリストを作成
    dayR = D2R[d]

    # 荷物のリストの部分集合を列挙
    for n in range(len(dayR) + 1):
        for tarR in combinations(dayR, n):
            # 要件(iii):荷物の重量の確認
            w = sum([R2W[r] for r in tarR])
            if w > 4000:
                continue

            # 荷物の配送先のお店を抽出
            tarS = set([R2S[r] for r in tarR])
            tarStp = tuple(sorted(tarS))

            # 要件(iv):実行可能な配送ルートとの照合
            if tarStp in Stp2Pat:
                X, time = Stp2Pat[tarStp]
                pat = (X, time, list(tarR))
                D2Pat[d].append(pat)
            else:
                continue
    print(f"配送日:{d} 配送パターン数:{len(D2Pat[d])}")

# %% [markdown]
# ### ②配送パターンを利用した数理モデルの実装・実験・検証

# %%
# 配送日に、配送可能な配送パターンのリストを紐づける辞書
D2Q = {d: [] for d in D}

# 配送パターンに、配送可能な荷物のリストを紐づける辞書
Q2R = {}

# 配送パターンに、移動時間を紐づける辞書
Q2T = {}

# 配送パターンに、残業時間を紐づける辞書
Q2H = {}

# 配送パターンに、配送ルートを紐づける辞書
Q2X = {}

for d in D:
    for q_no, (X, time, tarR) in enumerate(D2Pat[d]):
        q = f"q_{d}_{q_no}"
        D2Q[d].append(q)
        Q2R[q] = tarR
        Q2T[q] = time
        Q2H[q] = max(time - 8, 0)
        Q2X[q] = X
print("配送日1日目の配送パターン:", D2Q[1])

# %%
q = "q_1_20"
print("対象となる配送パターンの識別子:", q)
print("荷物のリスト:", Q2R[q])
print("移動時間:", Q2T[q])
print("残業時間:", Q2H[q])
print("配送ルート:", Q2X[q])

# %% [markdown]
# ### ★「配送パターンを利用した数理モデル」の実装

# %%
import pulp


prob = pulp.LpProblem("vrp_enum", pulp.LpMinimize)

x = {}
for d in D:
    for q in D2Q[d]:
        x[d, q] = pulp.LpVariable(f"x_{d}_{q}", cat="Binary")
y = pulp.LpVariable.dicts("y", R, cat="Continuous", lowBound=0, upBound=1)

# (A') 各配送日に1つの配送パターンを選択
for d in D:
    prob += pulp.lpSum([x[d, q] for q in D2Q[d]]) == 1

# (B') 各荷物は外注するか自社配送のどちらかを選択
for r in R:
    prob += y[r] + pulp.lpSum(x[d, q] for d in D for q in D2Q[d] if r in Q2R[q]) == 1

# (C') 費用合計と移動時間合計の最小化
zangyo = pulp.lpSum([3000 * Q2H[q] * x[d, q] for d in D for q in D2Q[d]])
gaityu = pulp.lpSum([46 * R2W[r] * y[r] for r in R])
time = pulp.lpSum([x[d, q] * Q2T[q] for d in D for q in D2Q[d]])
prob += zangyo + gaityu + time

# 求解
solver = pulp.PULP_CBC_CMD(msg=1)
status = prob.solve(solver)
print("ステータス:", pulp.LpStatus[status])

# 結果の表示
print(f"残業費用:{zangyo.value():.0f}[円]")
print(f"外注費用:{gaityu.value():.0f}[円]")
print(f"費用剛健:{zangyo.value() + gaityu.value():.0f}[円]")
print(f"移動時間:{time.value():.0f}[時間]")

# %%
resD2Q = {d: q for d in D for q in D2Q[d] if x[d, q].value() == 1}
for d in D:
    tar_q = resD2Q[d]

    # 移動する地点の順番のリストを作成
    X = Q2X[tar_q].copy()
    tar = p
    Route = [p]
    while len(X) >= 1:
        for k1, k2 in X:
            if k1 == tar:
                tar = k2
                Route.append(k2)
                X.remove((k1, k2))
    print(f"---配送日:{d} 配送パターン:{tar_q}---")
    print(f"移動時間:{Q2T[tar_q]:.2f}[時間]")
    print(f"残業時間:{Q2H[tar_q]:.2f}[時間]")
    print("配送ルート:", "->".join(Route))
    for r in Q2R[tar_q]:
        print(f"荷物{r}-お店{R2S[r]}")

# %%
import matplotlib.pyplot as plt


# 各地点の座標の取得
K2XY = {row.k: (row.x, row.y) for row in k_df.itertuples()}

fig = plt.figure(figsize=(12, 20))
for i in range(len(D)):
    d = D[i]
    tar_q = resD2Q[d]
    X = Q2X[tar_q]
    routeK = [k1 for k1, k2 in X]
    time = Q2T[tar_q]
    title_text = f"day:{d}({time:.1f}[h])"

    ax = fig.add_subplot(
        5, 4, i + 1, xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), title=title_text
    )
    ax.set_aspect("equal")

    for row in k_df.itertuples():
        if row.k in routeK:
            if row.depo_flag:
                ax.scatter(row.x, row.y, marker="o")
            else:
                ax.scatter(row.x, row.y, marker="x")

    for k1, k2 in X:
        (x1, y1) = K2XY[k1]
        (x2, y2) = K2XY[k2]
        ax.arrow(
            x1, y1, (x2 - x1), (y2 - y1), head_width=0.2, length_includes_head=True
        )

# %%
