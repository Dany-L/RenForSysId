import numpy as np
from src.example.inverted_pendulum import InvertedPendulum
import matplotlib.pyplot as plt
import os
import torch
from scipy.optimize import fsolve

ROOT_FIG_FOLDER = os.path.join(os.getcwd(), 'slides', 'fig')
np.random.seed(2023)

if __name__== "__main__":

    model = InvertedPendulum(m=1,l=1,mu=0.9)
    N = 3
    u = [[[np.random.normal(0,0.1)]] for i in range(N)]
    y = model.simulate_open_loop(
        N = N, 
        x0=np.array([[0.5],[0]]), 
        u=u
    )

    n_z = 10
    n_x = N

    for restart in range(10):
        W_z = torch.nn.Linear(in_features=n_z, out_features=n_z, bias=True)
        torch.nn.init.normal_(W_z.weight)
        U_z = torch.nn.Linear(in_features=n_x, out_features=n_z, bias=False)
        W_y = torch.nn.Linear(in_features=n_z, out_features=n_x, bias=True)

        W_z_numpy = W_z.weight.detach().numpy()
        U_z_numpy = U_z.weight.detach().numpy()
        b_z_numpy = W_z.bias.detach().numpy().reshape(n_z,1)
        W_y_numpy = W_y.weight.detach().numpy()
        b_y_numpy = W_y.bias.detach().numpy().reshape(n_x, 1)

        # print(f'min EW of (I-W_z): {np.min(np.real(np.linalg.eig(np.eye(n_z) - W_z_numpy)[0])):.4}')

        # for L in range(40):
            # fixed layer sequence model

        L = 40
        nl = torch.tanh

        ## finite number of layers
        # initial z^0
        # forward pass for fixed number of layers
        z = torch.zeros(size=(1, n_z))
        x = torch.tensor(u).reshape(1, n_x)
        for l in range(L):
            z = nl(W_z(z) + U_z(x))
        y_hat = W_y(z)
            
        ## fixed point
        z_0 = np.zeros(shape=(n_z, 1))
        x = np.array(u).reshape(n_x,1)

        def g_theta(z):
            z = z.reshape(n_z,1)
            return np.squeeze(np.tanh(W_z_numpy @ z + U_z_numpy @ x + b_z_numpy) - z)

        z_star, infodict, ier, mesg = fsolve(g_theta, x0=z_0, full_output=True)
        z_star = z_star.reshape(n_z, 1)
        y_hat_eq = W_y_numpy @ z_star + b_y_numpy



        # print(f'min EW of (I-W_z): {np.min(np.real(np.linalg.eig(np.eye(n_z) - W_z_numpy)[0]))} \t Number of finite layer: {L} \t Error DEQ and finite layer network: {np.linalg.norm(z.detach().numpy().T - z_star)}')
        # if L < 5 or L % 10 == 0:
        print(f'min EW of (I-W_z): {np.min(np.real(np.linalg.eig(np.eye(n_z) - W_z_numpy)[0])):.4} \t L: {L} \t || z^L - z^* ||^2: {np.linalg.norm(z.detach().numpy().T - z_star):.4}')
        if ier > 1:
            print(mesg)




    # model.animate(
    #     N=N,
    #     x0 = np.array([[0.3], [-2.5]]),
    #     u=[np.array([[0]]) for i in range(N)],
    #     title='Nonlinear pendulum dynamics',
    #     filepath=os.path.join(ROOT_FIG_FOLDER, 'inv_pendulum.mov'),
    # )

    # K = np.array([[19.4061, 5.4387]])
    
    # t = np.linspace(0,(N-1)/100, N)
    # y = np.squeeze(y)
    # plt.figure()
    # plt.title('Phase')
    # plt.plot(y[:,0], y[:,1])
    # plt.xlabel('x_1')
    # plt.ylabel('x_2')

    # plt.figure()
    # plt.plot(t, y[:, 0])
    # plt.xlabel('t')
    # plt.ylabel('x_1')

    # plt.figure()
    # plt.plot(t, y[:, 1])
    # plt.xlabel('t')
    # plt.ylabel('x_2')
    
    # plt.show()




