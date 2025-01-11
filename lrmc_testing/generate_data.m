function [X_star, Y, omega_r, omega_c] = generate_data(d1, d2, r, alpha, p)
    % Generate synthetic data for LRMC
    % Inputs:
    % d1: number of rows
    % d2: number of columns
    % r: rank of the matrix
    % alpha: fraction of corrupted entries
    % p: fraction of observed entries

    % Outputs:
    % X_star: original matrix
    % Y: observed entries
    % omega_r: row indices of observed entries
    % omega_c: column indices of observed entries


    % Generate random matrices U and V
    U = randn(d1, r) / sqrt(d1);
    V = randn(d2, r) / sqrt(d2);
    X_star = U * V';

    % Generate the original matrix vectorized
    Y_vec = X_star(:); % Vectorize the matrix

    % Add noise to a fraction of the entries
    num_elements = d1 * d2;
    num_noise = floor(alpha * num_elements);
    idx_noise = randperm(num_elements, num_noise);

    % Mean absolute value for noise scaling
    a = mean(abs(Y_vec), 'all');
    S = 2 * a * (rand(length(idx_noise), 1) - 0.5);
    Y_vec(idx_noise) = Y_vec(idx_noise) + S;

    % Generate observation mask omega
    num_observed = floor(p * num_elements);
    idx_observed = randperm(num_elements, num_observed);

    % Initialize omega and set observed entries
    omega = zeros(d1, d2);
    omega(idx_observed) = 1;

    % Extract observed entries (non-zero entries of omega)
    [omega_r, omega_c] = find(omega);
    observed_entries = sub2ind([d1, d2], omega_r, omega_c);

    % Final observed matrix with noise applied only on observed entries
    Y = Y_vec(observed_entries);
end
