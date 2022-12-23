import numpy as np
from src.example.inverted_pendulum import InvertedPendulum
import matplotlib.pyplot as plt
import os

ROOT_FIG_FOLDER = os.path.join(os.getcwd(), 'slides', 'fig')

if __name__== "__main__":

    model = InvertedPendulum(m=1,l=1,mu=0.9)
    N = 1000
    u = np.array([[0]]) * N
    y = model.simulate_open_loop(
        N = N, 
        x0=np.array([[0.1],[0]]), 
        u=[np.array([[0]]) for i in range(N)]
    )

    model.animate(
        N=N,
        x0 = np.array([[0.3], [-2.5]]),
        u=[np.array([[0]]) for i in range(N)],
        title='Nonlinear pendulum dynamics',
        filepath=os.path.join(ROOT_FIG_FOLDER, 'inv_pendulum.mov'),
    )

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




