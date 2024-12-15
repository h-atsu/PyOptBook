# %% [markdown]
# # **【scipy.optimize.milp版】第2章 Python数理最適化チュートリアル**
#
# 本チュートリアルは`scipy==1.9.0`以降を前提に、`scipy.optimize.milp`の利用方法を解説します。
# なお、以下のリンクを参考にしています。
# https://scipy.github.io/devdocs/reference/generated/scipy.optimize.milp.html

# %% [markdown]
# ## **2.2 線形計画問題をPythonの数理最適化ライブラリで解く**
# まず、`pulp`を利用してモデリングした場合を確認します。

# %%
import pulp


problem = pulp.LpProblem("LP", pulp.LpMaximize)

x = pulp.LpVariable("x", cat="Continuous")
y = pulp.LpVariable("y", cat="Continuous")

problem += 1 * x + 3 * y <= 30
problem += 2 * x + 1 * y <= 40
problem += x >= 0
problem += y >= 0
problem.objective = x + 2 * y

status = problem.solve(pulp.PULP_CBC_CMD(msg=False))

print("Status:", pulp.LpStatus[status])
print("x=", x.value(), "y=", y.value(), "obj=", problem.objective.value())

# %% [markdown]
# ここで、`problem`を`print`することで定式化を確認することができます。

# %%
problem

# %% [markdown]
# 上記から最適化問題として、次のパラメータを持つことを確認できます。
#
# ```
# c = [1, 2]
# A = [[1, 3], [2, 1]]
# b_u = [30, 40]
# b_l = [-inf, -inf]
# ```
# パラメータを参考に、次のようにして `scipy`で線形計画問題を解くことができます。
# ただし、目的関数が最大化問題であることを考慮して`c`に`-1`を掛けていること、および変数`x`の定義域はデフォルトで`0 <= x <= np.ing`となることに注意してください。

# %%
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp


c = -np.array([1, 2])
A = np.array([[1, 3], [2, 1]])
b_u = np.array([30, 40])
b_l = np.array([-np.inf, -np.inf])

constraints = LinearConstraint(A, b_l, b_u)

res = milp(c=c, constraints=constraints)
print("x:", res.x)
print("obj=", -res.fun)

# %% [markdown]
# `pulp(CBC)`で解いた場合の次の結果と合致していることが確認できます。
# ```
# Status: Optimal
# x= 18.0 y= 4.0 obj= 26.0
# ```

# %% [markdown]
# ## **2.3 規模の大きな数理最適化問題をPythonの数理最適化ライブラリで解く**
# まず、`pulp`を利用してモデリングした場合を確認します。

# %%
import pandas as pd
import pulp


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
problem = pulp.LpProblem("LP2", pulp.LpMaximize)

# 変数の定義
x = pulp.LpVariable.dicts("x", P, cat="Continuous")

# 制約式の定義
for p in P:
    problem += x[p] >= 0
for m in M:
    problem += pulp.lpSum([require[p, m] * x[p] for p in P]) <= stock[m]

# 目的関数の定義
problem += pulp.lpSum([gain[p] * x[p] for p in P])

# 求解
status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
print("Status:", pulp.LpStatus[status])

# 計算結果の表示
for p in P:
    print(p, x[p].value())

print("obj=", problem.objective.value())

# %% [markdown]
# ここで、`problem`を`print`することで定式化を確認することができます。

# %%
problem

# %% [markdown]
# 上記から最適化問題として、次のパラメータを持つことを確認できます。
# ```
# c = [-3, -4, -4, -5]
# A = [[2, 3, 0, 2], [0, 2, 2, 2], [1, 0, 2, 2]]
# b_u = [35, 22, 27]
# b_l = [-inf, -inf, -inf]
# ```
# パラメータを参考に、次のようにして `scipy`で線形計画問題を解くことができます。
# ただし、目的関数が最大化問題であることを考慮して`c`に`-1`を掛けていること、および変数`x`の定義域はデフォルトで`0 <= x <= np.ing`となることに注意してください。

# %%
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize import milp


# 以下、arrayではなく、listで処理
c = [-3, -4, -4, -5]
A = [[2, 3, 0, 2], [0, 2, 2, 2], [1, 0, 2, 2]]
b_u = [35, 22, 27]
b_l = [-np.inf, -np.inf, -np.inf]

constraints = LinearConstraint(A, b_l, b_u)
res = milp(c=c, constraints=constraints)

print("x:", res.x)
print("obj=", -res.fun)

# %% [markdown]
# ## ファイルを利用する場合
# 本書のように入力ファイルのデータを用いる場合は次のようになります。

# %%
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint
from scipy.optimize import milp


# データの取得
require_df = pd.read_csv("../data/requires.csv")
stock_df = pd.read_csv("../data/stocks.csv")
gain_df = pd.read_csv("../data/gains.csv")

c = -gain_df.gain
b_u = stock_df.stock
b_l = [-np.inf] * len(b_u)

A = np.array(require_df.require).reshape(len(gain_df["p"]), len(stock_df["m"])).T
constraints = LinearConstraint(A, b_l, b_u)
res = milp(c=c, constraints=constraints)

print("x:", res.x)
print("obj=", -res.fun)

# %% [markdown]
# ## 整数計画問題
# 整数計画問題として問題を解く場合は次のようになります。

# %%
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint
from scipy.optimize import milp


# データの取得
require_df = pd.read_csv("../data/requires.csv")
stock_df = pd.read_csv("../data/stocks.csv")
gain_df = pd.read_csv("../data/gains.csv")

c = -gain_df.gain
b_u = stock_df.stock
b_l = [-np.inf] * len(b_u)

A = np.array(require_df.require).reshape(len(gain_df["p"]), len(stock_df["m"])).T
constraints = LinearConstraint(A, b_l, b_u)

integrality = np.ones_like(c)
res = milp(c=c, constraints=constraints, integrality=integrality)

print("x:", res.x)
print("obj=", -res.fun)

# %%
