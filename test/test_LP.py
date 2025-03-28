#%%
import numpy as np
import matplotlib.pyplot as plt
import sys, os

parent = os.path.join(sys.path[0], "../")
sys.path.append(parent)

import src.LP as LP
import src.QP as QP

print("-"*50)

# %%

c = np.array([-3, -2, 0, 0])
A_eq = np.array([
  [1, 1, 1, 0],
  [2, 1/2, 0 , 1]
])
b_eq = np.array([5, 8])
# %%
model = LP.LP(c, A_eq = A_eq, b_eq = b_eq)
# %%
x, lam, s_N, B =  model.simplex.one_step_simplex(B = [2, 3])
print(f"One step simplex: \n\t x = {x},\n\t lam = {lam},\n\t s_N = {s_N},\n\t B = {B}")

x, lam, s_N, x_arr = model.simplex.simplex()
print(f"Simplex till stop: \n\t x = {x},\n\t lam = {lam},\n\t s_N = {s_N}")
print("-"*50)

x, lam, s_N, x_arr = model.interior_point.primal_dual_path([1,1,1,1], [0,0], [0.2,0.2,0.2,0.2])
print(f"Primal dual path: \n\t x = {x},\n\t lam = {lam},\n\t s_N = {s_N}")

x, la, s_N, x_arr = model.interior_point.long_step_path([1,1,1,1], [0,0], [0.2,0.2,0.2,0.2])
print(f"Long step path: \n\t x = {x},\n\t lam = {lam},\n\t s_N = {s_N}")

x, la, s_N, x_arr = model.interior_point.predictor_corrector()
print(f"Predictor-Corrector: \n\t x = {x},\n\t lam = {lam},\n\t s_N = {s_N}")

G = np.zeros((2,2))
c = np.array([-2, -5])
A = np.array([[1, -2], [-1, -2], [-1, 2], [1, 0], [0, 1]])
b = np.array([-2, -6, -2, 0, 0])

tester_QP = QP.QP(G, c, A_ineq = A, b_ineq = b)

fig, ax = tester_QP.plot((-1, 4), (-1, 4))

x, xarr = tester_QP.dual_active_set()

print(f"True solution: {x}")

plt.scatter(x[0], x[1], color = "red")

lp = LP.LP(c, A_l = A, b_l = b)

x, la, s_N, x_arr = lp.simplex.simplex()
print(f"Simplex till stop: \n\t x = {x}")
