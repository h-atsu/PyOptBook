# %% [markdown]
# # **第2章 Python数理最適化チュートリアル**

# %% [markdown]
# ## **2.1 連立一次方程式をPythonの数理最適化ライブラリで解く**

# %% [markdown]
# ### 全体のコード

# %%
import mip


problem = mip.Model()

x = problem.add_var(lb=-mip.INF)
y = problem.add_var(lb=-mip.INF)

problem += 120 * x + 150 * y == 1440
problem += x + y == 10

problem.optimize()

print("Status:", problem.status)
print("x=", x.x, "y=", y.x)

#####
# import pulp
#
# problem = pulp.LpProblem('SLE', pulp.LpMaximize)
#
# x = pulp.LpVariable('x', cat='Continuous')
# y = pulp.LpVariable('y', cat='Continuous')
#
# problem += 120 * x + 150 * y == 1440
# problem += x + y == 10
#
# status = problem.solve()
#
# print('Status:', pulp.LpStatus[status])
# print('x=', x.value(), 'y=', y.value())

# %% [markdown]
# ### 本書における逐次実行

# %%
# Pythonライブラリpython-mipの取り込み
import mip


## PythonライブラリPuLPの取り込み
# import pulp

# %%
# 数理モデルの定義
problem = mip.Model()
# problem = pulp.LpProblem('SLE', pulp.LpMaximize)

# %%
# 変数の定義
x = problem.add_var("x", lb=-mip.INF)
y = problem.add_var("y", lb=-mip.INF)
# x = pulp.LpVariable('x', cat='Continuous')
# y = pulp.LpVariable('y', cat='Continuous')

# %%
# 制約式の定義(pulpとpython-mipで同じ)
problem += 120 * x + 150 * y == 1440
problem += x + y == 10

# %%
# 求解
problem.optimize()
print("Status:", problem.status)
#####
# status = problem.solve()
# print('Status:', pulp.LpStatus[status])

# %%
# 最適化結果の表示
print("x=", x.x, "y=", y.x)
# print('x=', x.value(), 'y=', y.value())

# %% [markdown]
# ## **2.2 線形計画問題をPythonの数理最適化ライブラリで解く**

# %% [markdown]
# ### 全体のコード

# %%
import mip


problem = mip.Model()

x = problem.add_var()  # >=0
y = problem.add_var()  # >=0

problem += 1 * x + 3 * y <= 30
problem += 2 * x + 1 * y <= 40
problem.objective = mip.maximize(x + 2 * y)

problem.optimize()

print("Status:", problem.status)
print("x=", x.x, "y=", y.x, "obj=", problem.objective.x)

#####
# import pulp
#
# problem = pulp.LpProblem('LP', pulp.LpMaximize)
#
# x = pulp.LpVariable('x', cat='Continuous')
# y = pulp.LpVariable('y', cat='Continuous')
#
# problem += 1 * x + 3 * y <= 30
# problem += 2 * x + 1 * y <= 40
# problem += x >= 0
# problem += y >= 0
# problem.objective = x + 2 * y
#
# status = problem.solve()
#
# print('Status:', pulp.LpStatus[status])
# print('x=', x.value(), 'y=', y.value(), 'obj=', problem.objective.value())

# %% [markdown]
# ### 本書における逐次実行

# %%
# Pythonライブラリpython-mipの取り込み
import mip


# PythonライブラリPuLPの取り込み
# import pulp

# %%
# 数理最適化モデルの定義
problem = mip.Model()
# problem = pulp.LpProblem('LP', pulp.LpMaximize)

# %%
# 変数の定義
x = problem.add_var("x")  # >=0
y = problem.add_var("y")  # >=0
# x = pulp.LpVariable('x', cat='Continuous')
# y = pulp.LpVariable('y', cat='Continuous')

# %%
# 制約式の定義
problem += 1 * x + 3 * y <= 30
problem += 2 * x + 1 * y <= 40

#####
# problem += 1 * x + 3 * y <= 30
# problem += 2 * x + 1 * y <= 40
# problem += x >= 0
# problem += y >= 0

# %%
# 目的関数の定義
problem.objective = mip.maximize(x + 2 * y)
# problem.objective = x + 2 * y

# %%
# 求解
problem.optimize()
print("Status:", problem.status)
# status = problem.solve()
# print('Status:', pulp.LpStatus[status])

# %%
# 最適化結果の表示
print("x=", x.x, "y=", y.x, "obj=", problem.objective.x)
# print('x=', x.value(), 'y=', y.value(), 'obj=', problem.objective.value())

# %% [markdown]
# ## **2.3 規模の大きな数理最適化問題をPythonの数理最適化ライブラリで解く**

# %% [markdown]
# ### **線形計画問題**

# %% [markdown]
# ### ①データのインポート

# %%
# データ処理のためのライブラリpandasとPythonライブラリPuLPの取り込み
import pandas as pd


# %%
# stocks.csvからのデータ取得
stock_df = pd.read_csv("../data/stocks.csv")
stock_df

# %%
# requires.csvからのデータ取得
require_df = pd.read_csv("../data/requires.csv")
require_df

# %%
# gains.csvからのデータ取得
gain_df = pd.read_csv("../data/gains.csv")
gain_df

# %% [markdown]
# ### ②リストの定義

# %%
# 製品のリストの定義
P = gain_df["p"].tolist()
P

# %%
# 原料のリストの定義
M = stock_df["m"].tolist()
M

# %% [markdown]
# ### ③定数の定義

# %%
# 定数の定義:stock
stock = {row.m: row.stock for row in stock_df.itertuples()}

# stock = dict(zip(stock_df['m'], stock_df['stock']))
# stock = dict((row.m, row.stock) for row in stock_df.itertuples())
# stock = {row['m']:row['stock'] for i, row in stock_df.iterrows()} # 追記:iterrowsは低速なので避ける
# stock = stock_df.set_index('m').to_dict()['stock']
# stock = stock_df.set_index('m')['stock'].to_dict() # 追記
stock

# %%
# 定数の定義:gain
gain = {row.p: row.gain for row in gain_df.itertuples()}
gain

# %%
# 定数の定義:require
require = {(row.p, row.m): row.require for row in require_df.itertuples()}
require

# %% [markdown]
# ### ④線形計画問題の定義

# %%
# 数理最適化モデルの定義
problem = mip.Model()
# problem = pulp.LpProblem('LP2', pulp.LpMaximize)

# %% [markdown]
# ### ⑤変数の定義

# %%
# 変数の定義
x = {p: problem.add_var() for p in P}

# 変数の逐次定義
# x = {}
# for p in P:
#    x[p] = problem.add_var()

#####
# x = pulp.LpVariable.dicts('x', P, cat='Continuous')

# 変数の逐次定義
# x = {}
# for p in P:
#    x[p] = pulp.LpVariable('x_{}'.format(p), cat='Continuous')

# f-strings(Python3.6以降)
# x = {}
# for p in P:
#    x[p] = pulp.LpVariable(f'x_{p}', cat='Continuous')

# 辞書 & f-strings
# x = {p:pulp.LpVariable(f'x_{p}', cat='Continuous') for p in P}

# %% [markdown]
# ### ⑥制約式の定義

# %%
# 生産量は0以上
for p in P:
    problem += x[p] >= 0

# %%
# 生産量は在庫の範囲
for m in M:
    problem += mip.xsum([require[p, m] * x[p] for p in P]) <= stock[m]
# for m in M:
#    problem += pulp.lpSum([require[p,m] * x[p] for p in P]) <= stock[m]

# %% [markdown]
# ### ⑦目的関数の定義

# %%
# 目的関数の定義
problem.objective = mip.maximize(mip.xsum([gain[p] * x[p] for p in P]))
# problem += pulp.lpSum([gain[p] * x[p] for p in P])

# %% [markdown]
# ### 実行

# %%
# 求解
problem.optimize()
print("Status:", problem.status)
# status = problem.solve()
# print('Status:', pulp.LpStatus[status])

# %%
# 計算結果の表示
for p in P:
    print(p, x[p].x)

print("obj=", problem.objective.x)

#####
# for p in P:
#    print(p, x[p].value())
#
# print('obj=', problem.objective.value())

# %% [markdown]
# ### ⑧実装した数理最適化モデルのまとめ

# %%
import mip
import pandas as pd


# データの取得
require_df = pd.read_csv("../data/requires.csv")
stock_df = pd.read_csv("../data/stocks.csv")
gain_df = pd.read_csv("../data/gains.csv")

# 集合の定義
P = gain_df["p"].tolist()
M = stock_df["m"].tolist()

# 定数の定義
stock = {row.m: row.stock for row in stock_df.itertuples()}
gain = {row.p: row.gain for row in gain_df.itertuples()}
require = {(row.p, row.m): row.require for row in require_df.itertuples()}

# 数理最適化モデルの定義
problem = mip.Model()

# 変数の定義
x = {p: problem.add_var() for p in P}

# 制約式の定義
for p in P:
    problem += x[p] >= 0
for m in M:
    problem += mip.xsum([require[p, m] * x[p] for p in P]) <= stock[m]

# 目的関数の定義
problem.objective = mip.maximize(mip.xsum([gain[p] * x[p] for p in P]))

# 求解
problem.optimize()
print("Status:", problem.status)

# 計算結果の表示
for p in P:
    print(p, x[p].x)

print("obj=", problem.objective.x)

#####
# import pandas as pd
# import pulp
#
## データの取得
# require_df = pd.read_csv('requires.csv')
# stock_df = pd.read_csv('stocks.csv')
# gain_df = pd.read_csv('gains.csv')
#
## 集合の定義
# P = gain_df['p'].tolist()
# M = stock_df['m'].tolist()
#
## 定数の定義
# stock = {row.m:row.stock for row in stock_df.itertuples()}
# gain = {row.p:row.gain for row in gain_df.itertuples()}
# require = {(row.p,row.m):row.require for row in require_df.itertuples()}
#
## 数理最適化モデルの定義
# problem = pulp.LpProblem('LP2', pulp.LpMaximize)
#
## 変数の定義
# x = pulp.LpVariable.dicts('x', P, cat='Continuous')
#
## 制約式の定義
# for p in P:
#    problem += x[p] >= 0
# for m in M:
#    problem += pulp.lpSum([require[p,m] * x[p] for p in P]) <= stock[m]
#
## 目的関数の定義
# problem += pulp.lpSum([gain[p] * x[p] for p in P])
#
## 求解
# status = problem.solve()
# print('Status:', pulp.LpStatus[status])
#
## 計算結果の表示
# for p in P:
#    print(p, x[p].value())
#
# print('obj=', problem.objective.value())

# %% [markdown]
# ### **整数計画問題**

# %% [markdown]
# ### コード全体

# %%
import mip
import pandas as pd


# データの取得
require_df = pd.read_csv("../data/requires.csv")
stock_df = pd.read_csv("../data/stocks.csv")
gain_df = pd.read_csv("../data/gains.csv")

# 集合の定義
P = gain_df["p"].tolist()
M = stock_df["m"].tolist()

# 定数の定義
stock = {row.m: row.stock for row in stock_df.itertuples()}
gain = {row.p: row.gain for row in gain_df.itertuples()}
require = {(row.p, row.m): row.require for row in require_df.itertuples()}

# 数理最適化モデルの定義
problem = mip.Model()

# 変数の定義
x = {p: problem.add_var(var_type="I") for p in P}  # 変更点

# 制約式の定義
for p in P:
    problem += x[p] >= 0
for m in M:
    problem += mip.xsum([require[p, m] * x[p] for p in P]) <= stock[m]

# 目的関数の定義
problem.objective = mip.maximize(mip.xsum([gain[p] * x[p] for p in P]))

# 求解
problem.optimize()
print("Status:", problem.status)

# 計算結果の表示
for p in P:
    print(p, x[p].x)

print("obj=", problem.objective.x)

#####
# import pandas as pd
# import pulp
#
## データの取得
# require_df = pd.read_csv('requires.csv')
# stock_df = pd.read_csv('stocks.csv')
# gain_df = pd.read_csv('gains.csv')
#
## 集合の定義
# P = gain_df['p'].tolist()
# M = stock_df['m'].tolist()
#
## 定数の定義
# stock = {row.m:row.stock for row in stock_df.itertuples()}
# gain = {row.p:row.gain for row in gain_df.itertuples()}
# require = {(row.p,row.m):row.require for row in require_df.itertuples()}
#
## 数理最適化モデルの定義
# problem = pulp.LpProblem('IP', pulp.LpMaximize) # 変更点（任意）
#
## 変数の定義
# x = pulp.LpVariable.dicts('x', P, cat='Integer') # 変更点
#
## 制約式の定義
# for p in P:
#    problem += x[p] >= 0
# for m in M:
#    problem += pulp.lpSum([require[p,m] * x[p] for p in P]) <= stock[m]
#
## 目的関数の定義
# problem += pulp.lpSum([gain[p] * x[p] for p in P])
#
## 求解
# status = problem.solve()
# print('Status:', pulp.LpStatus[status])
#
## 計算結果の表示
# for p in P:
#    print(p, x[p].value())
#
# print('obj=', problem.objective.value())

# %%
