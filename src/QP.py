import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from itertools import combinations

class QP:  
  def __init__(
    self, 
    G : np.ndarray, 
    c : np.ndarray,
    const : float = 0,
    A_eq : np.ndarray = NotImplemented,
    b_eq : np.ndarray = NotImplemented, 
    A_ineq : np.ndarray = NotImplemented, 
    b_ineq : np.ndarray = NotImplemented
    ):
    """
    Initializes the QP object.
    """
    self.N = G.shape[0]
    self.G = G
    self.c = c
    self.const = const
    if A_eq is NotImplemented:
      A_eq = np.zeros((0, self.N))
    self.A_eq = A_eq
    if b_eq is NotImplemented:
      b_eq = np.zeros(0)
    self.b_eq = b_eq
    if A_ineq is NotImplemented:
      A_ineq = np.zeros((0, self.N))
    self.A_ineq = A_ineq
    if b_ineq is NotImplemented:
      b_ineq = np.zeros(0)
    self.b_ineq = b_ineq
    self.KKT = np.block([[G, A_eq.T], [A_eq, np.zeros((A_eq.shape[0], A_eq.shape[0]))]])
    
  def q(self, x : float):
    return 0.5 * x.T @ self.G @ x + self.c.T @ x + self.const
  
  def constraints(self, x : float):
    c_eq = self.A_eq @ x - self.b_eq
    c_ineq = self.A_ineq @ x - self.b_ineq
    return c_eq, c_ineq
  
  def is_feasible(self, x : np.ndarray):
    c_eq, c_ineq = self.constraints(x)
    return np.all(np.abs(c_eq) < 1e-8) and np.all(c_ineq >= -1e-8)
    
  def plot(
    self,
    x1_interval : tuple = (-10,10),
    x2_interval : tuple = (-10,10),
    ):
    """
    Plots the quadratic function.
    Args:
      interval (tuple): The interval to plot the function over.
      x_center (float): The x-coordinate of the center of the plot.
      y_center (float): The y-
      coordinate of the center of the plot.
    """      
      
    x = np.linspace(x1_interval[0], x1_interval[1], 300)
    y = np.linspace(x2_interval[0], x2_interval[1], 300)
    
    X, Y = np.meshgrid(x, y)
    
    Z = np.array([[self.q(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])
    C = np.array([[self.is_feasible(np.array([x, y])) for x, y in zip(row_x, row_y)] for row_x, row_y in zip(X, Y)])

    
    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, 20, cmap='RdGy')
    ax.contourf(X, Y, C, 20, alpha=0.5, cmap='Grays_r')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Quadratic Function")
    fig.tight_layout()
    return fig, ax

  def solve_eq(
    self, 
    G = NotImplemented, 
    c = NotImplemented, 
    A = NotImplemented, 
    b = NotImplemented, 
    method : str = NotImplemented
    ):
    """
    Solves the equality constrained quadratic programming problem.
    This function constructs and solves a linear system of equations derived from the 
    Karush-Kuhn-Tucker (KKT) conditions for equality constrained quadratic programming.
    Returns:
      tuple: A tuple containing:
        - numpy.ndarray: The solution vector for the primal variables.
        - numpy.ndarray: The solution vector for the Lagrange multipliers.
    """
    if A is NotImplemented:
      A = self.A_eq
    if b is NotImplemented:
      b = self.b_eq
    if G is NotImplemented:
      G = self.G
    if c is NotImplemented:
      c = self.c
    
    M = np.block([[self.G, -A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    v = np.concatenate([-c, b])
    
    if method == "LU":
      p, l, u = la.lu(M)
      x = la.solve(u, la.solve(l, v))
      return x[:self.c.shape[0]], x[self.c.shape[0]:]
    if method == "QR":
      q, r = la.qr(M)
      x = la.solve(r, q.T @ v)
      return x[:self.c.shape[0]], x[self.c.shape[0]:]
    else:
      x = np.linalg.lstsq(M, v, rcond=None)[0]
      return x[:self.c.shape[0]], x[self.c.shape[0]:]

  def solve_working_set(
    self, 
    W, 
    G = NotImplemented, 
    c = NotImplemented, 
    A_eq = NotImplemented, 
    b_eq = NotImplemented, 
    A_ineq = NotImplemented,
    b_ineq = NotImplemented,
    turnoff_print = False
    ):
    """
    Solves the quadratic programming problem using the active set method.
    """
    if G is NotImplemented:
      G = self.G
    if c is NotImplemented:
      c = self.c
    if A_eq is NotImplemented:
      A_eq = self.A_eq
    if b_eq is NotImplemented:
      b_eq = self.b_eq
    if A_ineq is NotImplemented:
      A_ineq = self.A_ineq
    if b_ineq is NotImplemented:
      b_ineq = self.b_ineq
      
    try:
      A = np.block([[A_eq], [A_ineq[W]]])
      b = np.concatenate([b_eq, b_ineq[W]])
      
      res = self.solve_eq(
        G = G,
        c = c,
        A = A, 
        b = b
        )
      
      # print(W, A, b, res[0])
      
      return res
    except:
      if not turnoff_print:
        print("The working set is not feasible.")
        print(f" W = {W}")
      
  def solve_idiot_active_set(self):
    """
    Solves the quadratic programming problem using the idiot active set method.
    """
    
    x_best = np.zeros(self.N)
    lam_best = np.zeros(0)
    F = 1e8
    w_best = []
    W = [np.array(list(combinations(range(self.A_ineq.shape[0]), i + 1))) for i in range(self.A_ineq.shape[0])]
    for w in W:
      for w_i in w:
        try:
          x, lam = self.solve_working_set(w_i, turnoff_print = True)
          F_new = self.q(x)
          if F_new < F and np.all(self.is_feasible(x)):
            F = F_new
            x_best = x
            lam_best = lam
            w_best = w_i
        except:
          continue
            
    return x_best, lam_best, w_best
  
  def __alpha(self, x, p, W):
    """
    Computes the step size for the active set method.
    """
    W_d = np.delete(range(self.A_ineq.shape[0]), W)
    bi = np.delete(self.b_ineq, W)
    ai = np.delete(self.A_ineq, W, axis = 0)
    
    
    ap = np.dot(ai, p)
    frac = (bi - np.dot(ai, x)) / ap
    frac = frac[ap < 0]
    W_d = W_d[ap < 0]
    
    alpha = min(1, np.min(frac))
    
    W_d = W_d[frac < 1]
    frac = frac[frac < 1]
    
    if len(frac) == 0:
      W_new = []
    else:
      W_new = W_d[np.argmax(frac)]
    
    return alpha, W_new
  
  def primal_active_set(self, x_0, W_0, N_it = 1e3):
    """
    Solves the quadratic programming problem using the active set method.
    """
    x_arr = np.empty(0)
    gk = lambda x: self.G @ x + self.c
    
    x = x_0
    W = W_0
    pk = np.ones(self.N)
    
    x_arr = np.append(x_arr, [x])
    
    for _ in range(int(round(N_it,0))):
      W = np.sort(W)
      try:
        pk, _ = self.solve_working_set(W=W, b_eq = np.zeros_like(self.b_eq), b_ineq = np.zeros_like(self.b_ineq), c=gk(x), turnoff_print = True)
      except:
        print(f"The working set {W} is not feasible.")
        W = np.random.choice(range(self.A_ineq.shape[0]), 1)
        break
      if np.linalg.norm(pk) < 1e-8:
        ai = self.A_ineq[W]
        lam_i = np.linalg.lstsq(ai.T, gk(x), rcond=None)[0]
        if np.all(lam_i >= 0):
          return x, np.reshape(x_arr, (-1, self.N))
        else:
          i = np.argmin(lam_i)
          W = np.delete(W, i)
          continue
      else:
        alpha, W_new = self.__alpha(x, pk, W)
        x = x + alpha * pk
        
        x_arr = np.append(x_arr, [x])
        
        if alpha < 0.99:
          W = np.concatenate([W, [W_new]])
        else:
          continue
        
    return x, np.reshape(x_arr, (-1, self.N))
  
  def dual_active_set(self, N_it = 1e1):
    """
    Solves the quadratic programming problem using the active set method.
    """
    x_arr = np.empty(0)
    x = -np.linalg.pinv(self.G) @ self.c
    lam = np.zeros(self.A_ineq.shape[0] + self.A_eq.shape[0])
    W = np.empty(0, dtype=int)
    I_ = range(self.A_ineq.shape[0])
    
    x_arr = np.append(x_arr, [x])
    
    for _ in range(int(round(N_it, 0))):
      if self.is_feasible(x):
        return x
    _, c_ineq = self.constraints(x)
    r_arr = np.arange(c_ineq.shape[0])[c_ineq < 0]
    r = np.random.choice(r_arr)
    def cr(x):
      return self.A_ineq[r] @ x - self.b_ineq[r]
    while cr(x) < 0:
    # for _ in range(10):
      ai = np.block([[self.A_eq], [self.A_ineq[W]]])
      ar = self.A_ineq[r]
    
      M = np.block(
        [[self.G, -ai.T],
         [-ai, np.zeros((ai.shape[0], ai.shape[0]))]]
        )
      vec = np.concatenate([ar, np.zeros(ai.shape[0])])
      
      res = np.linalg.lstsq(M, vec, rcond=None)[0]
      p = res[:self.N]
      v = res[self.N:]
      
      if np.abs(ar @ p) < 1e-8:
        if np.all(v >= 0):
          ValueError("The problem is infesible.")
        else:
          j_arr = np.arange(v.shape[0])[v < 0]
          j = j_arr[np.argmin(-lam[j_arr] / v[j_arr])]
          t = -lam[j]/v[j]
          lam[W[j]] = lam[W[j]] + t * v[j]
          lam[r] = lam[r] + t
          W = np.delete(W, W[j])
      else:
        j_arr = np.arange(v.shape[0])[v < 0]
        if len(j_arr) == 0:
          t1 = 1e20
        else:
          j = j_arr[np.argmin(-lam[j_arr] / v[j_arr])]
          t1 = -lam[j]/v[j]
        t2 = -cr(x)/ (ar @ p)
        if t2 <= t1:
          x = x + t2*p
          
          x_arr = np.append(x_arr, [x])
          
          if j_arr != 0:
            lam[W[j]] = lam[W[j]] + t2 * v[j]
          lam[r] = lam[r] + t2
          W = np.concatenate([W, [r]])
        else:
          x = x + t1*p
          
          x_arr = np.append(x_arr, [x])
          
          if j_arr != 0:
            lam[W[j]] = lam[W[j]] + t1 * v[j]
          lam[r] = lam[r] + t1    
          W = np.delete(W, W[j])      
    return x, np.reshape(x_arr, (-1, self.N))
  
  def __make_lin_system(self, rl, ra, rc, rsz, s, z, sigma, mu):
    M = np.block([
      [
        self.G, 
        -self.A_eq.T, 
        -self.A_ineq.T, 
        np.zeros((self.N, self.A_ineq.shape[0]))
      ],
      [
        -self.A_eq, 
        np.zeros((self.A_eq.shape[0], self.A_eq.shape[0])), 
        np.zeros((self.A_eq.shape[0], self.A_ineq.shape[0])), 
        np.zeros((self.A_eq.shape[0], self.A_ineq.shape[0]))
      ],
      [
        -self.A_ineq, 
        np.zeros((self.A_ineq.shape[0], self.A_eq.shape[0])), 
        np.zeros((self.A_ineq.shape[0], self.A_ineq.shape[0])), 
        np.eye(self.A_ineq.shape[0])
      ],
      [
        np.zeros((self.A_ineq.shape[0], self.N)),
        np.zeros((self.A_ineq.shape[0], self.A_eq.shape[0])), 
        np.diag(s), 
        np.diag(z)
      ]
    ])
    
    vec = np.concatenate([
      -rl, 
      -ra, 
      -rc, 
      -rsz + sigma * mu * np.ones(self.A_ineq.shape[0])
    ])
    
    return M, vec
  
  def __get_residuals(self, x, y, z, s):
    rl = self.G @ x + self.c - self.A_eq.T @ y - self.A_ineq.T @ z
    ra = -self.A_eq @ x + self.b_eq
    rc = -self.A_ineq @ x + self.b_ineq + s
    rsz = np.diag(s) @ np.diag(z) @ np.ones(self.A_ineq.shape[0])
    
    return rl, ra, rc, rsz

  def __test_alpha(self, alpha, s, z, ds, dz):
        first = np.min(s + alpha * ds) > 0
        second = np.min(z + alpha * dz) > 0
        return first and second

  def __test_convergence(self, tol, rl, ra, rc, rsz):
    f = np.linalg.norm(rl)
    s = np.linalg.norm(ra)
    t = np.linalg.norm(rc)
    q = np.linalg.norm(rsz)
    return f < tol and s < tol and t < tol and q < tol * 1e-2
      
  def primal_dual_active_set(self, x_0, y_0, z_0, s_0, tol = 1e-6, N_it = 1e3):
    N_it = int(round(N_it, 0))
    
    x = np.array(x_0)
    y = np.array(y_0)
    s = np.array(s_0)
    z = np.array(z_0)
    
    if np.all(np.block([z_0, s_0]) > 0):
      ValueError("s_0 and z_0 must be positive.")
    
    x_arr = np.empty(0)
    x_arr = np.append(x_arr, [x])
    
    for i in range(N_it):
      mu = (s@z)/self.A_ineq.shape[0]
      sigma = 10**(-i)
      
      rl, ra, rc, rsz = self.__get_residuals(x, y, z, s)
      
      M, vec = self.__make_lin_system(rl, ra, rc, rsz, s, z, sigma, mu)

      res = np.linalg.lstsq(M, vec, rcond=None)[0]
      dx, dy, dz, ds = res[:self.N], res[self.N:(self.N + self.A_eq.shape[0])], res[(self.N + self.A_eq.shape[0]):(self.N + self.A_eq.shape[0] + self.A_ineq.shape[0])], res[(self.N + self.A_eq.shape[0] + self.A_ineq.shape[0]):]
    
      alpha = 1
      while not self.__test_alpha(alpha, s, z, ds, dz):
        alpha = alpha*.9
      
      x = x + alpha * dx
      y = y + alpha * dy
      z = z + alpha * dz
      s = s + alpha * ds
      
      x_arr = np.append(x_arr, [x])

      if self.__test_convergence(tol, rl, ra, rc, rsz):
        break
      
    return x, np.reshape(x_arr, (-1, self.N))
  
  def predictor_corr(self, x_0, y_0, z_0, s_0, tol = 1e-6, N_it = 1e3):
    N_it = int(round(N_it, 0))
    
    x = np.array(x_0)
    y = np.array(y_0)
    s = np.array(s_0)
    z = np.array(z_0)
    
    if np.all(np.block([z_0, s_0]) > 0):
      ValueError("s_0 and z_0 must be positive.")
      
    x_arr = np.empty(0)
    x_arr = np.append(x_arr, [x])
    
    for i in range(N_it):
      
      rl, ra, rc, rsz = self.__get_residuals(x, y, z, s)
      
      M, vec = self.__make_lin_system(rl, ra, rc, rsz, s, z, 0, 0)
      
      res = np.linalg.lstsq(M, vec, rcond=None)[0]
      _, _, dz_aff, ds_aff = res[:self.N], res[self.N:(self.N + self.A_eq.shape[0])], res[(self.N + self.A_eq.shape[0]):(self.N + self.A_eq.shape[0] + self.A_ineq.shape[0])], res[(self.N + self.A_eq.shape[0] + self.A_ineq.shape[0]):]
      
      alpha_aff = 1
      while not self.__test_alpha(alpha_aff, s, z, ds_aff, dz_aff):
        alpha_aff = alpha_aff*.9
      
      mu_aff = (s + alpha_aff * ds_aff) @ (z + alpha_aff * dz_aff) / self.A_ineq.shape[0]
      mu = (s @ z) / self.A_ineq.shape[0]
      
      sigma = (mu_aff / mu)**3
      
      rsz_bar = rsz - np.diag(ds_aff) @ np.diag(dz_aff) @ np.ones(self.A_ineq.shape[0])
      
      M, vec = self.__make_lin_system(rl, ra, rc, rsz_bar, s, z, sigma, mu)
      res = np.linalg.lstsq(M, vec, rcond=None)[0]
      
      dx, dy, dz, ds = res[:self.N], res[self.N:(self.N + self.A_eq.shape[0])], res[(self.N + self.A_eq.shape[0]):(self.N + self.A_eq.shape[0] + self.A_ineq.shape[0])], res[(self.N + self.A_eq.shape[0] + self.A_ineq.shape[0]):]
      
      alpha = 1
      while not self.__test_alpha(alpha, s, z, ds, dz):
        alpha = alpha*.9
      
      eta = 0.995
      
      x = x + eta * alpha * dx
      y = y + eta * alpha * dy
      z = z + eta * alpha * dz
      s = s + eta * alpha * ds
      
      x_arr = np.append(x_arr, [x])
      
      if self.__test_convergence(tol, rl, ra, rc, rsz):
        break
      
    return x, np.reshape(x_arr, (-1, self.N))
  
  def primal_dual_predictor_corr(self, x_0, y_0, z_0, s_0, epsilon = 1e-6):
    x_0 = np.array(x_0)
    y_0 = np.array(y_0)
    z_0 = np.array(z_0)
    s_0 = np.array(s_0)
    
    if np.all(np.block([z_0, s_0]) > 0):
      ValueError("s_0 and z_0 must be positive.")
      
    x_arr = np.empty(0)
    x, y, z, s = x_0, y_0, z_0, s_0
    
    mc = z.shape[0]
    
    x_arr = np.append(x_arr, [x])
   
    rl = self.G @ x + self.c - self.A_eq.T @ y - self.A_ineq.T @ z
    ra = -self.A_eq @ x + self.b_eq
    rc = -self.A_ineq @ x + self.b_ineq + s
    rsz = np.diag(s) @ np.diag(z) @ np.ones(mc)
    mu = (z @ s)/mc
    
    while True:
      G_bar = self.G + self.A_ineq.T @ (np.diag(1/s)@np.diag(z)) @ self.A_ineq
      
      rl_bar = rl - self.A_ineq.T @ (np.diag(1/s) @ np.diag(z)) @ (rc - np.diag(1/z) @ rsz)
      
      M1 = np.block([
        [G_bar, -self.A_eq.T],
        [-self.A_eq, np.zeros((self.A_eq.shape[0], self.A_eq.shape[0]))]
      ])
      vec1 = np.block([rl_bar, ra])
      
      res = np.linalg.lstsq(M1, -vec1, rcond=None)[0]
      dx_aff, dy_aff = res[:self.N], res[self.N:]
      
      dz_aff = (np.diag(1/s) @ np.diag(z)) @ (-self.A_ineq @ dx_aff + rc - np.diag(1/z) @ rsz)
      ds_aff = -np.diag(1/z) @ (rsz + np.diag(s) @ dz_aff)

      alpha = 1
      while not self.__test_alpha(alpha, s, z, ds_aff, dz_aff):
        alpha = alpha*.9
      
      
      mu_aff = (z + alpha*dz_aff) @ (s + alpha*ds_aff)/mc
      sigma = (mu_aff/mu)**3
      
      rsz_bar = rsz + np.diag(ds_aff) @ np.diag(dz_aff) @ np.ones(mc) - sigma*mu * np.ones(mc)
      
      rl_bar = rl - self.A_ineq.T @ (np.diag(1/s) @ np.diag(z)) @ (rc - np.diag(1/z) @ rsz_bar)
      
      M2 = np.block([
        [G_bar, -self.A_eq.T],
        [-self.A_eq, np.zeros((self.A_eq.shape[0], self.A_eq.shape[0]))]
      ])
      
      vec2 = np.block([rl_bar, ra])
      
      res = np.linalg.lstsq(M2, -vec2, rcond=None)[0]
      dx, dy = res[:self.N], res[self.N:]
      
      dz = (np.diag(1/s) @ np.diag(z)) @ (-self.A_ineq @ dx + rc - np.diag(1/z) @ rsz_bar)
      ds = (-np.diag(1/z)) @ (rsz_bar + np.diag(s) @ dz)
      
      alpha = 1
      while not self.__test_alpha(alpha, s, z, ds, dz):
        alpha = alpha*.9
      
      eta = 0.995
      alpha_bar = eta * alpha
      x = x + alpha_bar * dx
      y = y + alpha_bar * dy
      z = z + alpha_bar * dz
      s = s + alpha_bar * ds
      
      rl = self.G @ x + self.c - self.A_eq.T @ y - self.A_ineq.T @ z
      ra = -self.A_eq @ x + self.b_eq
      rc = -self.A_ineq @ x + self.b_ineq + s
      rsz = np.diag(s) @ np.diag(z) @ np.ones(mc)
      
      mu = (s @ z)/mc
      
      x_arr = np.append(x_arr, [x])
      if np.linalg.norm(rl) < epsilon and np.linalg.norm(ra) < epsilon and np.linalg.norm(rc) < epsilon and np.linalg.norm(rsz) < epsilon:
        break
      
    return x, np.reshape(x_arr, (-1, self.N))