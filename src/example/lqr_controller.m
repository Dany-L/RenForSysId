l = 1; g=9.81;m=1;delta=0.01;mu=0.9;
% linearized pendulum
A = [1, delta; (g*delta)/l 1-(mu*delta)/(m*l^2)];
B = [0; delta/(m*l^2)];
C = [1 0];
D = 0;

sys = ss(A,B,C,D, delta);
[K,S,CLP] = dlqr(sys.A, sys.B,eye(2),1);
K

