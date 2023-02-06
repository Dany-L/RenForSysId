import numpy as np
from numpy.typing import NDArray
from ..models.dynamic_model import DynamicModel
import matplotlib.animation as anim
import matplotlib.pyplot as plt

class InvertedPendulum(DynamicModel):

    def __init__(
        self, 
        l: np.float64, 
        m: np.float64, 
        mu: np.float64, 
        delta:np.float64 = 0.01, 
        g: np.float64 = 9.81
    ) -> None:
        self.nu = 1
        self.ny = 1
        self.nx = 2
        self.nz = 1
        self.nw = 1
        super().__init__(
            A = np.array([[1, delta], [(g*delta)/l, 1 - (mu*delta)/(m * l**2)]]),
            B_1 = np.array([[0],[delta/(m*l**2)]]),
            B_2 = np.array([[0], [-(g*delta)/l]]),
            C_1 = np.array([[1, 0]]),
            C_2 = np.array([[1, 0]]),
            D_11 = np.zeros((self.ny, self.nu)),
            D_12 = np.zeros((self.ny, self.nw)),
            D_21 = np.zeros((self.nz, self.nu)),
            D_22 = np.array([[0.5]]),
            Delta=lambda z: z - np.sin(z)
        )
        self.m = m
        self.l = l
        self.g = g
        self.delta = delta
        self.mu = mu


    def animate(
        self, 
        N:int, 
        x0:NDArray[np.float64], 
        u:NDArray[np.float64],
        filepath: str,
        title: str = "",
        K: NDArray[np.float64] = None
    ) -> None:

        # generate data
        if K is None:
            y = np.squeeze(self.simulate_open_loop(
                N=N,
                x0=x0,
                u=u
            ))
        else:
            y = np.squeeze(self.simulate_closed_loop(
                N=N,
                x0=x0,
                K=K
            ))
        # get position values
        pos_x = self.l * np.sin(y[:, 0])
        pos_y = -self.l * np.cos(y[:, 0])

        # initialize figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1,1,1, 
            xlim=(-2, 2), 
            ylim=(-2, 2), 
            autoscale_on=False, 
            title=title
        )
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        # plot rod
        line, = ax.plot([], [], 'o-', lw=3.5)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def animate_init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate_step(k):
            thisx = [0, pos_x[k]]
            thisy = [0, -pos_y[k]]
            t = np.linspace(0, self.delta*k, k)

            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (k*self.delta))
            return line, time_text

        ani = anim.FuncAnimation(
            fig=fig, 
            func=animate_step, 
            frames=np.arange(0, len(y)-1),
            interval=1, 
            blit=True, 
            init_func=animate_init
        )
        ani.save(filename=filepath, fps=120)