import numpy as np
import matplotlib.pyplot as plt

class LP:
  
  
  def __init__(self, c, A_eq = NotImplemented, b_eq = NotImplemented, A_l = NotImplemented, b_l = NotImplemented, A_u = NotImplemented, b_u = NotImplemented, l = NotImplemented, u = NotImplemented):
    
    self.is_eq = True
    
    self.c = np.array(c)
    if A_eq is NotImplemented:
      self.A_eq = np.zeros((0, c.shape[0]))
    else:
      self.A_eq = np.array(A_eq)
    if b_eq is NotImplemented:
      self.b_eq = np.zeros(0)
    else:
      self.b_eq = np.array(b_eq)
    if A_l is NotImplemented:
      self.A_l = np.zeros((0, c.shape[0]))
    else:
      self.A_l = np.array(A_l)
      self.is_eq = False
    if b_l is NotImplemented:
      self.b_l = np.zeros(0)
    else:
      self.b_l = np.array(b_l)
    if A_u is NotImplemented:
      self.A_u = np.zeros((0, c.shape[0]))
    else:
      self.A_u = np.array(A_u)
      self.is_eq = False
    if b_u is NotImplemented:
      self.b_u = np.zeros(0)
    else:
      self.b_u = np.array(b_u)
    if l is NotImplemented:
      l = np.zeros(c.shape[0])
    
    if self.is_eq:
      self.slack_l = 0
      self.slack_u = 0
      self.n = self.c.shape[0]
      
      self.A = self.A_eq
      self.b = self.b_eq      
      
    else:
      self.slack_l = self.b_l.shape[0]
      self.slack_u = self.b_u.shape[0]
      self.n = self.c.shape[0]
      
      self.c = np.concatenate([
        -self.c,
        self.c,
        np.zeros(self.slack_l),
        np.zeros(self.slack_u),
      ])
      
      self.A = np.block([
        [-self.A_eq, self.A_eq, np.zeros((self.A_eq.shape[0],self.slack_l)), np.zeros((self.A_eq.shape[0], self.slack_u))],
        [-self.A_l, self.A_l, -np.eye(self.slack_l), np.zeros((self.slack_l, self.slack_u))],
        [-self.A_u, self.A_u, np.zeros((self.slack_u, self.slack_l)), np.eye(self.slack_u)],
      ])

      self.b = np.concatenate([
        self.b_eq,
        self.b_l,
        self.b_u,
      ])
    
    self.interior_point = self.Interior_point(self)
    self.simplex = self.Simplex(self)
    
  def dual(self):
    return LP(
      c = -self.b,
      A_eq = np.concatenate([self.A_eq.T, self.A_ineq.T], axis=0),
      b_eq = self.c
    )
  
  # def is_KKT(self, x, lam, s):
  #   first =  self.A.T @ lam + s = self.c
  #   second = self.A @ x - self.b = 0
  #   third = x >= 0
  #   fourth = s >= 0
  #   fith = s @ x = 0
    
  #   return first.all() and second.all() and third.all() and fourth.all() and fith.all()
  class Simplex:
    def __init__(self, parent):
      self.parent = parent
    
    def one_step_simplex(self, B, steps = 1, ineq_modified = False):
      B = np.array(B)
      print(ineq_modified, self.parent.is_eq)
      if not ineq_modified and not self.parent.is_eq:
        print("Hello")
        B = np.concatenate([B, B + self.parent.n])
      
      N = np.delete(np.arange(self.parent.c.shape[0]), B)
      print(B, N)
      A_B = self.parent.A[:, B]
      A_N = self.parent.A[:, N]
      c_B = self.parent.c[B]
      c_N = self.parent.c[N]
      x_B, _, _, _ = np.linalg.lstsq(A_B, self.parent.b)
      print(x_B)
      x_N = np.zeros(len(N))
      
      lam, _, _, _ = np.linalg.lstsq(A_B.T, c_B)
      
      s_N = c_N - A_N.T @ lam
      
      if np.all(s_N >= 0):
        x = np.zeros(self.parent.c.shape[0])
        x[B] = x_B
        x[N] = x_N
        return x, lam, s_N, B
      
      q = np.where(s_N < 0)[0][0]
      d, _, _, _ = np.linalg.lstsq(A_B, A_N[:, q])
      
      print(d)

      if np.all(d <= 0):
        raise ValueError("The problem is Unbounded")
      
      i_list = np.where(d > 0)[0]
      x_qp = np.min(x_B[i_list] / d[i_list])
      p = np.where(x_B / d == x_qp)[0][0]
      x_B = x_B - x_qp * d
      x_N[q] = x_qp
      
      x = np.zeros(self.parent.c.shape[0])
      x[B] = x_B
      x[N] = x_N
      
      p_add = N[q]
      q_add = B[p]
      
      B = np.append(B, p_add)
      N = np.append(N, q_add)
      
      B = np.delete(B, p)
      N = np.delete(N, q)
      
      B = np.sort(B)
      N = np.sort(N)
      
      if steps == 1:
        return x, lam, s_N, N
      return self.one_step_simplex(B, steps - 1, ineq_modified = True)
      
    def simplex(self):
      B = [0,1]
      x_arr = np.array([])
      while True:
        x, lam, s, B = self.one_step_simplex(B)
        B = B[B < self.parent.n]
        x_arr = np.append(x_arr, x)
        if np.all(s >= 0):
          return x, lam, s, x_arr
  
  class Interior_point:
    
    def __init__(self, parent):
      self.parent = parent
      
    def solve_scale(self, x, lam, s, sigma = 0, dx = NotImplemented, ds = NotImplemented):
      
      M = np.block([
          [np.zeros((self.parent.A.shape[1], self.parent.A.shape[1])), self.parent.A.T, np.eye(self.parent.A.shape[1])],
          [self.parent.A, np.zeros((self.parent.A.shape[0], self.parent.A.shape[0])), np.zeros((self.parent.A.shape[0], self.parent.A.shape[1]))],
          [np.diag(s), np.zeros((x.shape[0], self.parent.A.shape[0])), np.diag(x)]
        ])
        
      rb = self.parent.A @ x - self.parent.b
      rc = self.parent.A.T @ lam + s - self.parent.c
      mu = (x @ s) / x.shape[0]
      
      if dx is NotImplemented or ds is NotImplemented:
        vec = np.concatenate([-rc, -rb, -np.diag(x) @ np.diag(s) @ np.ones_like(x) + sigma * mu * np.ones_like(x)])
      else:
        vec = np.concatenate([
          -rc, 
          -rb, 
          -np.diag(x) @ np.diag(s) @ np.ones_like(x) - np.diag(dx) @ np.diag(ds) @ np.ones_like(x) + sigma * mu * np.ones_like(x)
        ])
      
      Delta, _, _, _ = np.linalg.lstsq(M, vec, rcond=-1)
      dx = Delta[:x.shape[0]]
      dlam = Delta[x.shape[0]:x.shape[0] + lam.shape[0]]
      ds = Delta[x.shape[0] + lam.shape[0]:]
      return dx, dlam, ds
    
    def primal_dual_path(self, x0, lam0, s0, it = 1e2, sigma = 0.5):
      x0 = np.array(x0)
      lam0 = np.array(lam0)
      s0 = np.array(s0)
      
      it = int(round(it,0))
      if not (np.all(x0 > 0) and np.all(s0 > 0)):
        raise ValueError("x0 and s0 must be strictly positive")
      
      x = x0
      lam = lam0
      s = s0
      
      x_arr = np.zeros((it + 1 , x0.shape[0]))
      x_arr[0] = x0
      
      for k in range(it):
        dx, dlam, ds = self.solve_scale(x, lam, s, sigma)
        
        alpha = 1
        while not (np.all(x + alpha * dx > 0) and np.all(s + alpha * ds > 0)):
          alpha *= 0.5
          
        x = x + alpha * dx
        x_arr[k+1] = x
        
        lam = lam + alpha * dlam
        s = s + alpha * ds
        
      return x, lam, s, x_arr
    
    def long_step_path(self, x0, lam0, s0, gamma = 0.5, sigma_int = (0.1,0.9), it = 1e2):
      sigma_min = sigma_int[0]
      sigma_max = sigma_int[1]
      
      x0 = np.array(x0)
      lam0 = np.array(lam0)
      s0 = np.array(s0)
      
      it = int(round(it,0))
      
      def in_N(x, lam, s):
        mu = (x @ s) / x.shape[0]
        return np.all(x @ s >= gamma * mu)
      
      if not (0 < sigma_min < sigma_max < 1):
        raise ValueError("0 < sigma_min < sigma_max < 1 must be satisfied")
      if not in_N(x0, lam0, s0):
        raise ValueError("Initial point must be in N_-inf")
      
      x = x0
      lam = lam0
      s = s0
      x_arr = np.zeros((it + 1 , x0.shape[0]))
      x_arr[0] = x0
      
      for k in range(it):
        sigma = sigma_min + (sigma_max - sigma_min) * k / it
        
        dx, dlam, ds = self.solve_scale(x, lam, s, sigma)
        
        alpha = 1
        while not in_N(x + alpha * dx, lam + alpha * dlam, s + alpha * ds):
          alpha *= 0.5
        
        x = x + alpha * dx
        x_arr[k+1] = x
        
        lam = lam + alpha * dlam
        s = s + alpha * ds
      return x, lam, s, x_arr
    
    def get_initial_point(self):
      x_tilde = self.parent.A.T @ np.linalg.inv(self.parent.A @ self.parent.A.T) @ self.parent.b
      lam_tilde = np.linalg.inv(self.parent.A @ self.parent.A.T) @ self.parent.A @ self.parent.c
      s_tilde = self.parent.c - self.parent.A.T @ lam_tilde
      dx = max(-3/2*np.max(x_tilde), 0)
      ds = max(-3/2*np.max(s_tilde), 0)
      
      x_hat = x_tilde + dx*np.ones_like(x_tilde)
      s_hat = s_tilde + ds*np.ones_like(s_tilde)
      dx_hat = x_hat @ s_hat / (2* np.ones_like(x_hat)@s_hat)
      ds_hat = x_hat @ s_hat / (2* np.ones_like(x_hat)@x_hat)
      
      x0 = x_hat + dx_hat*np.ones_like(x_hat)
      lam0 = lam_tilde
      s0 = s_hat + ds_hat*np.ones_like(s_hat)
      
      return x0, lam0, s0
    
    def predictor_corrector(self, it = 1e2):
      it = int(round(it,0))
      
      x0, lam0, s0 = self.get_initial_point()
      
      x = x0
      lam = lam0
      s = s0
      
      x_arr = np.zeros((it + 1, x0.shape[0]))
      
      for k in range(it):
        dx_aff, dlam_aff, ds_aff = self.solve_scale(x, lam, s, 0)
        
        i_list = np.where(dx_aff < 0)
        alpha_aff_pri = min(1, np.min(-x[i_list]/dx_aff[i_list]))
        
        i_list = np.where(ds_aff < 0)
        alpha_aff_dual = min(1, np.min(-s[i_list]/ds_aff[i_list]))
        
        mu_aff = (x + alpha_aff_pri * dx_aff) @ (s + alpha_aff_dual *ds_aff) / x.shape[0]
        mu = x @ s / x.shape[0]
        
        sigma = (mu_aff/mu)**3
        
        dx, dlam, ds = self.solve_scale(x, lam, s, sigma, dx = dx_aff, ds = ds_aff)
        
        eta = 0.9 + 0.1 * k/it
        
        i_list = np.where(dx < 0)
        alpha_pri = min(1, eta*np.min(-x[i_list]/dx[i_list]))
        
        i_list = np.where(ds < 0)
        alpha_dual = min(1, eta*np.min(-s[i_list]/ds[i_list]))
        
        x = x + alpha_pri * dx
        lam = lam + alpha_dual * dlam
        s = s + alpha_dual * ds
        
        x_arr[k+1] = x
      return x, lam, s, x_arr