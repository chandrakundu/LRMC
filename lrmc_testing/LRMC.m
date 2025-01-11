function [L,R] = LRMC(Y,n1,n2,r,p,omega_row,omega_col,etas,zetas)  
    % Learned Robust Matrix Completion (LRMC) Algorithm
    % Inputs:
    % Y: observed entries
    % n1: number of rows
    % n2: number of columns
    % r: rank of the matrix
    % p: fraction of observed entries
    % omega_row: row indices of observed entries
    % omega_col: column indices of observed entries
    % etas: trained step parameters  
    % zetas: trained threshold parameters

    % Outputs:
    % L: left factorized matrix
    % R: right factorized matrix
        
    % preparation    
    num_iter = length(zetas);
    time_counter = 0;         

    % Initialization
    tstart = tic;
    if p == 1
        Y = sparse(omega_row, omega_col, Y);
        X0 = Y - soft_thres(Y, zetas(1));
    elseif p < 1
        X0 = (Y-soft_thres(Y,zetas(1)))*etas(1);  
        X0 = sparse(omega_row, omega_col, X0);
    else
        fprintf("p should be less than 1\n");
        return;
    end

    [U0,Sigma0,V0] = lansvd(X0,r);
    sqrt_sigma = sqrt(Sigma0);
    L = U0*sqrt_sigma;
    R = V0*sqrt_sigma; 
    tEnd = toc(tstart);
    time_counter = time_counter + tEnd;
    X_prev = sparse(omega_row, omega_col, Y);
    error = norm(L*R' - X_prev, 'fro')/norm(X_prev, 'fro'); 
    X_prev = L*R';  
    fprintf("===============LRMC logs=============\n");
    fprintf("Initialization error: %f, time: %f\n", error, time_counter);

    % Loop
    for k=2:num_iter
        tstart = tic;
        if p == 1            
            X = L * R';
            S = soft_thres(Y - X, zetas(k));
            Lk = L - etas(k) * (X + S - Y) * R / (R' * R + eps('double') * eye(r));
            Rk = R - etas(k) * (X + S - Y)' * L / (L' * L + eps('double') * eye(r));
        else
            diff = Y - XonOmega(L,R,omega_row,omega_col);       
            midterm = etas(k)*(soft_thres(diff, zetas(k)) - diff);  
            Lk = L-sparse_mult(midterm, omega_row, omega_col, R, n1, n2, r)/(R'*R);
            Rk = R-sparse_mult(midterm, omega_col, omega_row, L, n2, n1, r)/(L'*L);
        end
        L = Lk;
        R = Rk;
        tEnd = toc(tstart);
        time_counter = time_counter + tEnd;        
        error = norm(L*R' - X_prev, 'fro')/norm(X_prev, 'fro');    
        X_prev = L*R';   
        
        fprintf("iter: %d, error: %f, time: %f\n", k-1, error, time_counter);        
    end
    fprintf("======================================\n");
end

% Soft Thresholding Function
function S = soft_thres(S, zeta)
    S = sign(S).*max(abs(S)-zeta,0.0);
end
