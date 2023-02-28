import numpy as np
from numpy.typing import NDArray
from ..models.dynamic_model import DynamicModel
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import symbols, diff, sin, cos, sign, init_printing


init_printing(use_unicode=True)

class CartPoleLin(DynamicModel):
    def __init__(
        self, 
        A,
        B
    ) -> None:
        self.n_x = 4
        self.n_u = 1
        self.n_y = 2
        self.n_z = self.n_x
        super().__init__(
            A = A,
            B_1 = B,
            B_2 = np.zeros(shape=(self.n_x, self.n_z)), 
            C_1 = np.array([
                [1,0,0,0],
                [0,0,1,0]
            ]), 
            C_2 = np.zeros(shape=(self.n_z, self.n_x)), 
            D_11 = np.zeros(shape=(self.n_y, self.n_u)), 
            D_12 = np.zeros(shape=(self.n_y, self.n_z)), 
            D_21 = np.zeros(shape=(self.n_z, self.n_u)), 
            Delta = lambda z: z, 
            D_22 = np.zeros(shape=(self.n_z, self.n_z))
        )

class CartPoleNonlin():
    def __init__(
        self,
        tau
    ) -> None:
        self.n_x = 4
        self.n_u = 1
        self.n_y = 2
        self.n_z = self.n_x

        self.g = 9.8
        self.m_c = 1.0
        self.m_p = 0.1
        self.total_mass = (self.m_p + self.m_c)
        self.l = 0.5 # actually half the pole's length
        self.tau = tau  # seconds between state updates
        self.mu_c = 0.0 # coefficient of friction of cart on track
        self.mu_p = 0.1 # coefficient of friction of pole on cart
        self.C = np.array([
                [1,0,0,0],
                [0,0,1,0]
            ])

    def dynamics(self, t, x, u):
        u_k = u[int(t/self.tau)]
        x1, x2, x3, x4 = x
        x_dot = np.zeros_like(x)
        x_dot[3] = (self.g * np.sin(x3) + np.cos(x3) * ((- u_k - self.m_p * self.l * x4**2 * np.sin(x3) + self.mu_c * np.sign(x2)) / (self.m_c + self.m_p)) - (self.mu_p * x4) / (self.m_p * self.l)) / (self.l * (4 / 3 - (self.m_p * np.cos(x3)**2) / (self.m_c + self.m_p)))
        x_dot[0] = x2
        x_dot[1] = (u_k + self.m_p * self.l * x4**2 * np.sin(x3) - self.m_p * self.l * x_dot[3] * np.cos(x3) - self.mu_c * np.sign(x2)) / (self.m_c + self.m_p)
        x_dot[2] = x4
        return x_dot
    
    def simulate(self, u, T):
        sol = solve_ivp(fun=lambda t, y:self.dynamics(t, y, u), t_span=[0, T-self.tau], t_eval=np.linspace(0,T-self.tau,int(T*(1/self.tau))), y0=self.state)
        x = sol.y.T
        y = np.hstack([self.C @ x_k.reshape(self.n_x, 1) for x_k in x])
        return y

    def init(self, x0 = None):
        if x0:
            x, x_dot, theta, theta_dot = x0
        else:
            x = np.random.uniform(low=-0.5, high=0.5)
            x_dot = np.random.uniform(low=-0.05, high=0.05)
            theta = np.random.uniform(low=-3.14, high=3.14)
            theta_dot = np.random.uniform(low=-3, high=3)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        return self.state
    
    def symb_lin(self, linearization_point):
        # symbolic linearization evaluated at linearization point
        x1, x2, x3, x4, u = symbols('x1 x2 x3 x4 u')

        eval_dict = {x1: linearization_point[0], x2: linearization_point[1], x3:linearization_point[2], x4:linearization_point[3], u:0}
        
        x4_dot = (self.g * sin(x3) + cos(x3) * ((- u - self.m_p * self.l * x4**2 * sin(x3) + self.mu_c * sign(x2)) / (self.m_c + self.m_p)) - (self.mu_p * x4) / (self.m_p * self.l)) / (self.l * (4 / 3 - (self.m_p * cos(x3)**2) / (self.m_c + self.m_p)))
        x1_dot = x2
        x2_dot = (u + self.m_p * self.l * x4**2 * sin(x3) - self.m_p * self.l * x4_dot * cos(x3) - self.mu_c * sign(x2)) / (self.m_c + self.m_p)
        x3_dot = x4

        A = np.array([
            [diff(x1_dot, x1).evalf(subs=eval_dict), diff(x1_dot, x2).evalf(subs=eval_dict), diff(x1_dot, x3).evalf(subs=eval_dict), diff(x1_dot, x4).evalf(subs=eval_dict)],
            [diff(x2_dot, x1).evalf(subs=eval_dict), diff(x2_dot, x2).evalf(subs=eval_dict), diff(x2_dot, x3).evalf(subs=eval_dict), diff(x2_dot, x4).evalf(subs=eval_dict)],
            [diff(x3_dot, x1).evalf(subs=eval_dict), diff(x3_dot, x2).evalf(subs=eval_dict), diff(x3_dot, x3).evalf(subs=eval_dict), diff(x3_dot, x4).evalf(subs=eval_dict)],
            [diff(x4_dot, x1).evalf(subs=eval_dict), diff(x4_dot, x2).evalf(subs=eval_dict), diff(x4_dot, x3).evalf(subs=eval_dict), diff(x4_dot, x4).evalf(subs=eval_dict)],
        ])

        B = np.array([
            [diff(x1_dot, u).evalf(subs=eval_dict)],
            [diff(x2_dot, u).evalf(subs=eval_dict)],
            [diff(x3_dot, u).evalf(subs=eval_dict)],
            [diff(x4_dot, u).evalf(subs=eval_dict)],
        ])

        return A.astype(np.float32),B.astype(np.float32)
    