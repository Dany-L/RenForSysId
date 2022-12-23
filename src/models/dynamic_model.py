import numpy as np
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from typing import Tuple, List
from collections.abc import Callable
from scipy.optimize import fsolve

class DynamicModel(metaclass=ABCMeta):

    def __init__(
        self, 
        A:NDArray[np.float64], 
        B_1:NDArray[np.float64], 
        B_2:NDArray[np.float64], 
        C_1:NDArray[np.float64],
        C_2:NDArray[np.float64],
        D_11:NDArray[np.float64],
        D_12:NDArray[np.float64],
        D_21:NDArray[np.float64],
        Delta:Callable,
        D_22:NDArray[np.float64] = None,
    ) -> None:
        self.A = A
        self.B_1 = B_1
        self.B_2 = B_2
        self.C_1 = C_1
        self.C_2 = C_2
        self.D_11 = D_11
        self.D_12 = D_12
        self.D_21 = D_21
        self.D_22 = D_22

        self.nx, self.nu = self.B_1.shape
        _, self.nw = self.B_2.shape
        self.ny, _ = self.C_1.shape
        self.nz, _ = self.C_2.shape

        self.Delta = Delta

    def simulate_open_loop(
        self,
        N: int,
        x0: NDArray[np.float64],
        u: List[NDArray[np.float64]],
    ) -> List[NDArray[np.float64]]:

        x = x0
        y = []
        for k in range(N):
            # z = self.C_2 @ x + self.D_21 @ u[k] + self.D_22 @ w
            # w = self.Delta(z)
            w_star = fsolve(lambda w: np.squeeze(self.Delta(self.C_2 @ x + self.D_21 @ u[k] + self.D_22 @ w) - w), x0 = 0).reshape(self.nw, 1)
            x = self.A @ x + self.B_1 @ u[k] + self.B_2 @ w_star
            y.append(self.C_1 @ x + self.D_11 @ u[k] + self.D_12 @ w_star)
            
        return y

    def simulate_closed_loop(
        self,
        N: int,
        x0: NDArray[np.float64],
        K: NDArray[np.float64]
    ) -> List[NDArray[np.float64]]:
        x = x0
        y = []
        for k in range(N):
            # z = self.C_2 @ x + self.D_21 @ u[k] + self.D_22 @ w
            # w = self.Delta(z)
            u = - np.clip(K @ x, -5,5)
            w_star = fsolve(lambda w: np.squeeze(self.Delta(self.C_2 @ x + self.D_21 @ u + self.D_22 @ w) - w), x0 = 0).reshape(self.nw, 1)
            x = self.A @ x + self.B_1 @ u + self.B_2 @ w_star
            y.append(self.C_1 @ x + self.D_11 @ u + self.D_12 @ w_star)
            
        return y




