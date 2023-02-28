clear all, close all

% Shit sector bounds from [alpha beta] to [-1 1]
n_z = 1;
alpha = 0; % lower bound
beta = 1; % upper bound
% sector bounded nonlinearity: alpha \leq delta(x)/x \leq beta
% delta = @(z) max(z,0);
delta = @(z) tanh(z);
S_delta = 1/2*(alpha+beta);
L_delta = 1/2*(beta-alpha);

delta_tilde =@(z) L_delta^(-1) * (delta(z) - S_delta * z);

N = 100;
z = linspace(-5,5, N);

% plot
figure()
plot(z, delta(z), 'LineWidth',2), hold on, grid on
plot(z, delta_tilde(z), 'LineWidth',2)
legend('$w(z)$', '$\tilde{w}(z)$', 'interpreter', 'latex', 'fontsize', 16);
xlabel('$z$', 'interpreter', 'latex', 'FontSize',16);
plot(z, -z, 'Color','black', 'LineStyle','--', 'LineWidth',2)
plot(z, z, 'Color','black', 'LineStyle','--', 'LineWidth',2)


% test new sector bound
Lambda = eye(n_z);
P = [-2*Lambda (alpha+beta)*Lambda;...
    (alpha+beta)*Lambda -2*alpha*beta*Lambda];
P_tilde = [-Lambda 0; ...
    0 Lambda];

% plot
figure()
v_tilde = zeros(N,1);v = zeros(N,1);
for idx = 1:length(z)
    v(idx) = [delta(z(idx));z(idx)]'*P*[delta(z(idx));z(idx)];
    v_tilde(idx) = [delta_tilde(z(idx));z(idx)]'*P_tilde*[delta_tilde(z(idx));z(idx)];
end
plot(z, v, 'LineWidth',2), hold on, grid on
plot(z, v_tilde, 'LineWidth',2)
legend('$[\Delta(z),~z] P [\Delta(z),~z]^{T}$', ...
    '$[\tilde{\Delta}(z),~z] \tilde{P} [\tilde{\Delta}(z),~z]^{T}$', ...
    'interpreter', 'latex', 'fontsize', 16);
xlabel('$z$', 'interpreter', 'latex', 'FontSize',18);

