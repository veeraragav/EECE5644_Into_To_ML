clc
clear

%% =============== True GMM parameters ==================== %%
d = 2;
alpha_true = [0.27, 0.26, 0.24, 0.23];
% mu_true(:,1) = [2;1];
% mu_true(:,2) = [4;3];
% mu_true(:,3) = [6;6];
% mu_true(:,4) = [8.5;8];
% Sigma_true(:,:,1) = [1, -0.9; -0.9, 1];
% Sigma_true(:,:,2) = [1.5, 0; 0, 0.5];
% Sigma_true(:,:,3) = [1, 0.9; 0.9, 1]; 
% Sigma_true(:,:,4) = [0.5, 0;0, 2];
mu_true(:,1) = [2.5;1.5];
mu_true(:,2) = [4.5;3.5];
mu_true(:,3) = [6;6];
mu_true(:,4) = [8.5;8];
Sigma_true(:,:,1) = [0.5, -0.2; -0.2, 0.5];
Sigma_true(:,:,2) = [0.5, 0; 0, 0.5];
Sigma_true(:,:,3) = [0.5, 0.2; 0.2, 0.5]; 
Sigma_true(:,:,4) = [0.5, 0;0, 0.1];

%% ============== Generating data from True GMM ============ %%
% D10 = randGMM(10,alpha_true,mu_true,Sigma_true);
% D100 = randGMM(100,alpha_true,mu_true,Sigma_true);
% D1000 = randGMM(1000,alpha_true,mu_true,Sigma_true);
%% ================ Demo ============================= %%
% D_train_1000 = datasample(D1000, 1500, 2);
% D_validate_1000 = datasample(D1000, 1500, 2);
% % x = randGMM(500,alpha_true,mu_true,Sigma_true);
% % [alpha_est, mu_est, sigma_est] = EMforGMM(D_train_1000, 2, 5)
% % evaluateGMM(D_validate_1000, alpha_est, mu_est, sigma_est)
% GMModel = fitgmdist(D_train_1000',4);
% alpha_est = GMModel.ComponentProportion';
% mu_est = GMModel.mu';
% sigma_est = GMModel.Sigma;
% evaluateGMM(D_validate_1000, alpha_est, mu_est, sigma_est)
% % figure(11);scatter(D_validate_1000(1,:), D_validate_1000(2,:));
%% ============== Analysis using D10 =============== %%
fprintf("\nD10 Analysis started\n");
selected_model_D10 = [];
for it = 1:1:100
    D10 = randGMM(10,alpha_true,mu_true,Sigma_true);
    B = 10;
    for b = 1:B
        D_train_50 = datasample(D10, 50, 2);
        D_validate_50 = datasample(D10, 50, 2);
        p_val_array = zeros(1,6); %stores performance of validation data over GMM of different orders
        for M = 1:1:6
            GMModel = fitgmdist(D_train_50', M, 'RegularizationValue', 0.1);
            alpha_est = GMModel.ComponentProportion';
            mu_est = GMModel.mu';
            sigma_est = GMModel.Sigma;
            p_val_array(M) = p_val_array(M) + evaluateGMM(D_validate_50, alpha_est, mu_est, sigma_est);
        end
    end
    p_val_array = p_val_array./B;
    [p_val_max, model_order] = max(p_val_array);
    selected_model_D10 = [selected_model_D10; model_order]; 
end

%% ================ Plot Results ================== %%
figure(2);
X = 1:6;
Y = [sum(selected_model_D10==1) sum(selected_model_D10==2) sum(selected_model_D10==3) sum(selected_model_D10==4) sum(selected_model_D10==5) sum(selected_model_D10==6)];
bar(X, Y, 0.4);
xlabel('GMM Order');
ylabel('No. of Selected Models');
title('D10');

%% ============== Analysis using D100 =============== %%
fprintf("\nD100 Analysis started\n");
selected_model_D100 = [];
for it = 1:1:100
    D100 = randGMM(100,alpha_true,mu_true,Sigma_true);
    B = 10;
    for b = 1:B
        D_train_300 = datasample(D100, 300, 2);
        D_validate_300 = datasample(D100, 300, 2);
        p_val_array = zeros(1,6); %stores performance of validation data over GMM of different orders
        for M = 1:1:6
            GMModel = fitgmdist(D_train_300', M, 'RegularizationValue', 0.1);
            alpha_est = GMModel.ComponentProportion';
            mu_est = GMModel.mu';
            sigma_est = GMModel.Sigma;
            p_val_array(M) = p_val_array(M) + evaluateGMM(D_validate_300, alpha_est, mu_est, sigma_est);
        end   
    end
    p_val_array = p_val_array./B;
    [p_val_max, model_order] = max(p_val_array);
    selected_model_D100 = [selected_model_D100; model_order]; 
end

%% ================ Plot Results ================== %%
figure(3);
X = 1:6;
Y = [sum(selected_model_D100==1) sum(selected_model_D100==2) sum(selected_model_D100==3) sum(selected_model_D100==4) sum(selected_model_D100==5) sum(selected_model_D100==6)];
bar(X, Y, 0.4);
xlabel('GMM Order');
ylabel('No. of Selected Models');
title('D100');

%% ============== Analysis using D1000 =============== %%
fprintf("\nD1000 Analysis started\n");
selected_model_D1000 = [];
for it = 1:1:100
    D1000 = randGMM(1000,alpha_true,mu_true,Sigma_true);
    B = 10;
    for b = 1:B
        D_train_2000 = datasample(D1000, 2000, 2);
        D_validate_2000 = datasample(D1000, 2000, 2);
        p_val_array = zeros(1,6); %stores performance of validation data over GMM of different orders
        for M = 1:1:6 
            GMModel = fitgmdist(D_train_2000', M, 'RegularizationValue', 0.1);
            alpha_est = GMModel.ComponentProportion';
            mu_est = GMModel.mu';
            sigma_est = GMModel.Sigma;
            p_val_array(M) = p_val_array(M) + evaluateGMM(D_validate_2000, alpha_est, mu_est, sigma_est);
        end
    end
    p_val_array = p_val_array./B;
    [p_val_max, model_order] = max(p_val_array);
    selected_model_D1000 = [selected_model_D1000; model_order]; 
end

%% ================ Plot Results ================== %%
figure(4);
X = 1:6;
Y = [sum(selected_model_D1000==1) sum(selected_model_D1000==2) sum(selected_model_D1000==3) sum(selected_model_D1000==4) sum(selected_model_D1000==5) sum(selected_model_D1000==6) ];
bar(X, Y, 0.4);
xlabel('GMM Order');
ylabel('No. of Selected Models');
title('D1000');

%% ===================== All Functions reside here =========================== %%
function p = evaluateGMM(x, alpha, mu, Sigma)
    N = size(x, 2);  
    M = size(alpha,1);
    d = size(mu, 1);
%     fprintf("\nEvaluating GMM of order %d with %d samples..\n", M, N);
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    p_x = sum(temp, 1);
    p = mean(log(p_x));
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end
%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end


