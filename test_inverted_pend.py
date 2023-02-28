import numpy as np
from src.example.cartpole import CartPoleLin, CartPoleNonlin
import matplotlib.pyplot as plt
import os
import torch
from scipy.optimize import fsolve
from scipy.signal import cont2discrete, lti, dlti, dstep

ROOT_FIG_FOLDER = os.path.join(os.getcwd(), 'slides', 'fig')
np.random.seed(2023)

if __name__== "__main__":
    T = 10
    tau = 0.02
    N = T*int(1/tau)
    # u = np.random.uniform(low=-0.1, high=0.1, size=(N))
    u = np.zeros(shape=(N,))
    model = CartPoleNonlin(tau=tau)
    x0 = model.init(x0=[0,0,np.pi+0.1,0])
    y_non_lin = model.simulate(T=T, u=u)

    A, B = model.symb_lin(linearization_point=[0,0,np.pi,0])
    C = (model.C).astype(np.float32)
    D = np.zeros(shape=(model.n_y, 1), dtype=np.float32)
    sys_c = lti(A,B,C,D)
    A_d, B_d, C_d, D_d, _ = cont2discrete((A,B,C,D), dt=tau)

    print(f'lambda(A): {np.real(np.linalg.eig(A_d)[0])}')

    model = CartPoleLin(A_d,B_d)
    y_lin = model.simulate_open_loop(x0=x0.reshape(model.nx, 1), u=u.tolist(), N = N)
    y_lin = np.hstack(y_lin)

    t = np.linspace(0,T-tau,N)
    

    def plot_states(k):
        plt.figure()
        plt.plot(t, y_non_lin[k,:], label='non lin')
        # plt.plot(t, y_lin[k,:], label='lin')
        plt.xlabel('time')
        plt.ylabel(f'y_{k+1}')
        plt.legend()

    plot_states(0)
    plot_states(1)

    plt.show()


    print(f'Error between linear and nonlinear model: {np.linalg.norm(y_lin- y_non_lin)}')



