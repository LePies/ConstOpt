#%%
import numpy as np
import matplotlib.pyplot as plt
import sys, os

parent = os.path.join(sys.path[0], "../")
sys.path.append(parent)

import src.QP as QP
#%%
G = np.array([[2, 0], [0, 2]])
c = np.array([-2, -5])
const = 1 + 2.5**2
A = np.array([[1, -2], [-1, -2], [-1, 2], [1, 0], [0, 1]])
b = np.array([-2, -6, -2, 0, 0])

#%%
model_ineq = QP.QP(
  G = G,
  c = c,
  A_ineq = A,
  b_ineq = b,
  const=const
)

model_eq = QP.QP(
  G = G,
  c = c,
  A_eq = np.array([[1, 1]]),
  b_eq = np.array([3])
)

print(f"Starndard solver solution: {model_eq.solve_eq()[0]}")
print(f"LU solver solution: {model_eq.solve_eq('LU')[0]}")
print(f"QR solver solution: {model_eq.solve_eq('QR')[0]}")

x, _ = model_ineq.solve_working_set(W = [0])
print(f"Working set solver solution: {x}")

x, _, _ = model_ineq.solve_idiot_active_set()
print(f"Idiot active set solver solution: {x}")


fig, ax = model_ineq.plot((-0.5,5),(-0.5,5))
x, x_arr = model_ineq.primal_active_set(W_0 = [3], x_0 = [0,0])
print(f"Primal active set solver solution: {x}")
plt.plot(x_arr[:,0], x_arr[:,1], ls="--", marker="o", color="black", label="Primal active set")

x, x_arr = model_ineq.dual_active_set()
print(f"Dual active set solver solution: {x}")

plt.plot(x_arr[:,0], x_arr[:,1], ls="--", marker="x", color="red", label="Dual active set")


x, x_arr = model_ineq.primal_dual_active_set(x_0=[1,0], y_0=[], z_0=[1,1,1,1,1], s_0=[1,1,1,1,1])
print(f"Primal Dual solver solution: {x}")
plt.plot(x_arr[:,0], x_arr[:,1], ls="--", marker="x", color="yellow", label="Primal dual")

x, x_arr = model_ineq.predictor_corr(x_0=[0.1,1], y_0=[], z_0=[1,1,1,1,1], s_0=[1,1,1,1,1])
print(f"Predictor Corrector solver solution: {x}")
plt.plot(x_arr[:,0], x_arr[:,1], ls="--", marker="x", color="gray", label="Predictor Corrector")

# print(x_arr)

x, x_arr = model_ineq.primal_dual_predictor_corr(x_0=[2,1], y_0=[], z_0=[1,1,1,1,1], s_0=[1,1,1,1,1])
print(f"Primal dual Predictor Corrector active set solver solution: {x}")
plt.plot(x_arr[:,0], x_arr[:,1], ls="--", marker="x", color="blue", label="Primal dual - Predictor corrector")
plt.legend()
plt.xlim(-0.5,5)
plt.ylim(-0.5,5)
plt.show()


# %%
x = np.linspace(-0.5, 5, 100)
plt.plot(x,(x+2)/2, label = "0")
plt.plot(x,(-x+6)/2, label = "1")
plt.plot(x,-(-x+2)/2, label = "2")
plt.plot(x,0*x, label = "3")
plt.plot(0*x,x, label = "4")
plt.legend()
# %%
