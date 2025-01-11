%% Test LRMC on synthetic data
clear;
addpath PROPACK;
addpath trained_models;

%% Parameters
n = 3000;               % size of the matrix
n1 = n; 
n2 = n;
r = 5;                  % rank of the matrix
alpha = 0.1;            % fraction of corrupted entries
p = 0.1;                % fraction of observed entries

%% Generate data
[X_star, Y, omega_r, omega_c] = generate_data(n1, n2, r, alpha, p);

%% Run LRMC
model_path = strcat('lrmc_n3000_r5_alpha',num2str(alpha),'_p',num2str(p),'.mat');
load(model_path)

[L,R] = LRMC(Y,n1,n2,r,p,omega_r,omega_c,eta, zeta);


%% Compare with the original matrix
norm(L*R' - X_star, 'fro')/norm(X_star, 'fro')
