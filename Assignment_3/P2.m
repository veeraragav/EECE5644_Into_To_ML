clc
clear
%% ======================= Load Data ========================== %%
load('data/D100_train.mat');
load('data/D500_train.mat');
load('data/D1000_train.mat');
load('data/D10000_test.mat');
M_max = 6;
%% ============== Analysis using D100 =============== %%
K = 10;
n_samples = size(D100_train,1);
batch_size = n_samples / K;
c0_p = [];
c1_p = [];
c2_p =[];
for k = 1:1:K

    data_val = D100_train((k-1)*batch_size +1 : k*batch_size, :);
    if k == 1
        data_train = D100_train(k*batch_size+1:size(D100_train,1), :);

    elseif k == K
        data_train = D100_train(1:(k-1)*batch_size, :);

    else
        data_train1 = D100_train(1:(k - 1) * batch_size, :);
        data_train2 = D100_train(k * batch_size+1:size(D100_train,1), :);
        data_train = [data_train1; data_train2];
    end  
    
    c0_data_train = [];
    c1_data_train = [];
    c2_data_train = [];
    for it = 1:1:size(data_train, 1)
        if data_train(it, 1) == 1
            c0_data_train = [c0_data_train; data_train(it, 2:3)];
        end
        if data_train(it, 1) == 2
            c1_data_train = [c1_data_train; data_train(it, 2:3)];
        end
        if data_train(it, 1) == 3
            c2_data_train = [c2_data_train; data_train(it, 2:3)];
        end
    end
    
    c0_data_val = [];
    c1_data_val = [];
    c2_data_val = [];
    for it = 1:1:size(data_val, 1)
        if data_val(it, 1) == 1
            c0_data_val = [c0_data_val; data_val(it, 2:3)];
        end
        if data_val(it, 1) == 2
            c1_data_val = [c1_data_val; data_val(it, 2:3)];
        end
        if data_val(it, 1) == 3
            c2_data_val = [c2_data_val; data_val(it, 2:3)];
        end
    end
     
    options = statset('MaxIter',2000);
    for M = 1:1:M_max
        GMModel = fitgmdist(c0_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c0_p(k, M) = evaluateGMM(c0_data_val', alpha_est, mu_est, sigma_est);
    end
     
    for M = 1:1:M_max
        GMModel = fitgmdist(c1_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c1_p(k, M) = evaluateGMM(c1_data_val', alpha_est, mu_est, sigma_est);
    end
    
    for M = 1:1:M_max
        GMModel = fitgmdist(c2_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c2_p(k, M) = evaluateGMM(c2_data_val', alpha_est, mu_est, sigma_est);
    end
end
[p_val_max_c0, D100_model_order_c0] = max(mean(c0_p, 1));
[p_val_max_c1, D100_model_order_c1] = max(mean(c1_p, 1));
[p_val_max_c2, D100_model_order_c2] = max(mean(c2_p, 1));
%% ============== Analysis using D500 =============== %%
K = 10;
n_samples = size(D500_train,1);
batch_size = n_samples / K;
c0_p = [];
c1_p = [];
c2_p =[];
for k = 1:1:K

    data_val = D500_train((k-1)*batch_size +1 : k*batch_size, :);
    if k == 1
        data_train = D500_train(k*batch_size+1:size(D500_train,1), :);

    elseif k == K
        data_train = D500_train(1:(k-1)*batch_size, :);

    else
        data_train1 = D500_train(1:(k - 1) * batch_size, :);
        data_train2 = D500_train(k * batch_size+1:size(D500_train,1), :);
        data_train = [data_train1; data_train2];
    end  
    
    c0_data_train = [];
    c1_data_train = [];
    c2_data_train = [];
    for it = 1:1:size(data_train, 1)
        if data_train(it, 1) == 1
            c0_data_train = [c0_data_train; data_train(it, 2:3)];
        end
        if data_train(it, 1) == 2
            c1_data_train = [c1_data_train; data_train(it, 2:3)];
        end
        if data_train(it, 1) == 3
            c2_data_train = [c2_data_train; data_train(it, 2:3)];
        end
    end
    
    c0_data_val = [];
    c1_data_val = [];
    c2_data_val = [];
    for it = 1:1:size(data_val, 1)
        if data_val(it, 1) == 1
            c0_data_val = [c0_data_val; data_val(it, 2:3)];
        end
        if data_val(it, 1) == 2
            c1_data_val = [c1_data_val; data_val(it, 2:3)];
        end
        if data_val(it, 1) == 3
            c2_data_val = [c2_data_val; data_val(it, 2:3)];
        end
    end
     
    options = statset('MaxIter',2000);
    for M = 1:1:M_max
        GMModel = fitgmdist(c0_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c0_p(k, M) = evaluateGMM(c0_data_val', alpha_est, mu_est, sigma_est);
    end
     
    for M = 1:1:M_max
        GMModel = fitgmdist(c1_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c1_p(k, M) = evaluateGMM(c1_data_val', alpha_est, mu_est, sigma_est);
    end
    
    for M = 1:1:M_max
        GMModel = fitgmdist(c2_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c2_p(k, M) = evaluateGMM(c2_data_val', alpha_est, mu_est, sigma_est);
    end
end
[p_val_max_c0, D500_model_order_c0] = max(mean(c0_p, 1));
[p_val_max_c1, D500_model_order_c1] = max(mean(c1_p, 1));
[p_val_max_c2, D500_model_order_c2] = max(mean(c2_p, 1));

%% ============== Analysis using D1000 =============== %%
K = 10;
n_samples = size(D1000_train,1);
batch_size = n_samples / K;
c0_p = [];
c1_p = [];
c2_p =[];
for k = 1:1:K

    data_val = D1000_train((k-1)*batch_size +1 : k*batch_size, :);
    if k == 1
        data_train = D1000_train(k*batch_size+1:size(D1000_train,1), :);

    elseif k == K
        data_train = D1000_train(1:(k-1)*batch_size, :);

    else
        data_train1 = D1000_train(1:(k - 1) * batch_size, :);
        data_train2 = D1000_train(k * batch_size+1:size(D1000_train,1), :);
        data_train = [data_train1; data_train2];
    end  
    
    c0_data_train = [];
    c1_data_train = [];
    c2_data_train = [];
    for it = 1:1:size(data_train, 1)
        if data_train(it, 1) == 1
            c0_data_train = [c0_data_train; data_train(it, 2:3)];
        end
        if data_train(it, 1) == 2
            c1_data_train = [c1_data_train; data_train(it, 2:3)];
        end
        if data_train(it, 1) == 3
            c2_data_train = [c2_data_train; data_train(it, 2:3)];
        end
    end
    
    c0_data_val = [];
    c1_data_val = [];
    c2_data_val = [];
    for it = 1:1:size(data_val, 1)
        if data_val(it, 1) == 1
            c0_data_val = [c0_data_val; data_val(it, 2:3)];
        end
        if data_val(it, 1) == 2
            c1_data_val = [c1_data_val; data_val(it, 2:3)];
        end
        if data_val(it, 1) == 3
            c2_data_val = [c2_data_val; data_val(it, 2:3)];
        end
    end
     
    options = statset('MaxIter',2000);
    for M = 1:1:M_max
        GMModel = fitgmdist(c0_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c0_p(k, M) = evaluateGMM(c0_data_val', alpha_est, mu_est, sigma_est);
    end
     
    for M = 1:1:M_max
        GMModel = fitgmdist(c1_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c1_p(k, M) = evaluateGMM(c1_data_val', alpha_est, mu_est, sigma_est);
    end
    
    for M = 1:1:M_max
        GMModel = fitgmdist(c2_data_train, M,'Options',options, 'RegularizationValue', 0.1);
        alpha_est = GMModel.ComponentProportion';
        mu_est = GMModel.mu';
        sigma_est = GMModel.Sigma;
        c2_p(k, M) = evaluateGMM(c2_data_val', alpha_est, mu_est, sigma_est);
    end
end
[p_val_max_c0, D1000_model_order_c0] = max(mean(c0_p, 1));
[p_val_max_c1, D1000_model_order_c1] = max(mean(c1_p, 1));
[p_val_max_c2, D1000_model_order_c2] = max(mean(c2_p, 1));
%% ===================== GMMs from D100_train =================================== %%

c0_data_D100_train = [];
c1_data_D100_train = [];
c2_data_D100_train = [];
for it = 1:1:size(D100_train, 1)
    if D100_train(it, 1) == 1
        c0_data_D100_train = [c0_data_D100_train; D100_train(it, 2:3)];
    end
    if D100_train(it, 1) == 2
        c1_data_D100_train = [c1_data_D100_train; D100_train(it, 2:3)];
    end
    if D100_train(it, 1) == 3
        c2_data_D100_train = [c2_data_D100_train; D100_train(it, 2:3)];
    end
end

c0_gmm_D100  = fitgmdist(c0_data_D100_train, D100_model_order_c0,'Options',options, 'RegularizationValue', 0.1);
c1_gmm_D100  = fitgmdist(c1_data_D100_train, D100_model_order_c1,'Options',options, 'RegularizationValue', 0.1);
c2_gmm_D100  = fitgmdist(c2_data_D100_train, D100_model_order_c2,'Options',options, 'RegularizationValue', 0.1);

%% ===================== GMMs from D500_train =================================== %%

c0_data_D500_train = [];
c1_data_D500_train = [];
c2_data_D500_train = [];
for it = 1:1:size(D500_train, 1)
    if D500_train(it, 1) == 1
        c0_data_D500_train = [c0_data_D500_train; D500_train(it, 2:3)];
    end
    if D500_train(it, 1) == 2
        c1_data_D500_train = [c1_data_D500_train; D500_train(it, 2:3)];
    end
    if D500_train(it, 1) == 3
        c2_data_D500_train = [c2_data_D500_train; D500_train(it, 2:3)];
    end
end

c0_gmm_D500  = fitgmdist(c0_data_D500_train, D500_model_order_c0,'Options',options, 'RegularizationValue', 0.1);
c1_gmm_D500  = fitgmdist(c1_data_D500_train, D500_model_order_c1,'Options',options, 'RegularizationValue', 0.1);
c2_gmm_D500  = fitgmdist(c2_data_D500_train, D500_model_order_c2,'Options',options, 'RegularizationValue', 0.1);

%% ===================== GMMs from D1000_train =================================== %%

c0_data_D1000_train = [];
c1_data_D1000_train = [];
c2_data_D1000_train = [];
for it = 1:1:size(D1000_train, 1)
    if D1000_train(it, 1) == 1
        c0_data_D1000_train = [c0_data_D1000_train; D1000_train(it, 2:3)];
    end
    if D1000_train(it, 1) == 2
        c1_data_D1000_train = [c1_data_D1000_train; D1000_train(it, 2:3)];
    end
    if D1000_train(it, 1) == 3
        c2_data_D1000_train = [c2_data_D1000_train; D1000_train(it, 2:3)];
    end
end

c0_gmm_D1000  = fitgmdist(c0_data_D1000_train, D1000_model_order_c0,'Options',options, 'RegularizationValue', 0.1);
c1_gmm_D1000  = fitgmdist(c1_data_D1000_train, D1000_model_order_c1,'Options',options, 'RegularizationValue', 0.1);
c2_gmm_D1000  = fitgmdist(c2_data_D1000_train, D1000_model_order_c2,'Options',options, 'RegularizationValue', 0.1);

%% ================================= Dtest creation ============================= %%
c0_data_Dtest = [];
c1_data_Dtest = [];
c2_data_Dtest = [];
for it = 1:1:size(D10000_test, 1)
    if D10000_test(it, 1) == 1
        c0_data_Dtest = [c0_data_Dtest; D10000_test(it, 2:3)];
    end
    if D10000_test(it, 1) == 2
        c1_data_Dtest = [c1_data_Dtest; D10000_test(it, 2:3)];
    end
    if D10000_test(it, 1) == 3
        c2_data_Dtest = [c2_data_Dtest; D10000_test(it, 2:3)];
    end
end
%% ================================= MAP classifier from D100_train ============================= %%
prior_D100 = [size(c0_data_D100_train,1), size(c1_data_D100_train,1), size(c2_data_D100_train,1)] ./ size(D100_train, 1);

c0_classified_c0_D100 = [];
c0_classified_c1_D100 = [];
c0_classified_c2_D100 = [];

for it=1:1:size(c0_data_Dtest)
    p_x_0 = liklihood(c0_data_Dtest(it,:)', c0_gmm_D100.ComponentProportion', c0_gmm_D100.mu', c0_gmm_D100.Sigma);
    p_0 = p_x_0 * prior_D100(1);
    p_x_1 = liklihood(c0_data_Dtest(it,:)', c1_gmm_D100.ComponentProportion', c1_gmm_D100.mu', c1_gmm_D100.Sigma);
    p_1 = p_x_1 * prior_D100(2);
    p_x_2 = liklihood(c0_data_Dtest(it,:)', c2_gmm_D100.ComponentProportion', c2_gmm_D100.mu', c2_gmm_D100.Sigma);
    p_2 = p_x_2 * prior_D100(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c0_classified_c0_D100 = [c0_classified_c0_D100; c0_data_Dtest(it,:)];
    elseif class == 2
        c0_classified_c1_D100 = [c0_classified_c1_D100; c0_data_Dtest(it,:)];
    elseif class == 3
        c0_classified_c2_D100 = [c0_classified_c2_D100; c0_data_Dtest(it,:)];
    end
end

c1_classified_c0_D100 = [];
c1_classified_c1_D100 = [];
c1_classified_c2_D100 = [];
for it=1:1:size(c1_data_Dtest)
    p_x_0 = liklihood(c1_data_Dtest(it,:)', c0_gmm_D100.ComponentProportion', c0_gmm_D100.mu', c0_gmm_D100.Sigma);
    p_0 = p_x_0 * prior_D100(1);
    p_x_1 = liklihood(c1_data_Dtest(it,:)', c1_gmm_D100.ComponentProportion', c1_gmm_D100.mu', c1_gmm_D100.Sigma);
    p_1 = p_x_1 * prior_D100(2);
    p_x_2 = liklihood(c1_data_Dtest(it,:)', c2_gmm_D100.ComponentProportion', c2_gmm_D100.mu', c2_gmm_D100.Sigma);
    p_2 = p_x_2 * prior_D100(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c1_classified_c0_D100 = [c1_classified_c0_D100; c1_data_Dtest(it,:)];
    elseif class == 2
        c1_classified_c1_D100 = [c1_classified_c1_D100; c1_data_Dtest(it,:)];
    elseif class == 3
        c1_classified_c2_D100 = [c1_classified_c2_D100; c1_data_Dtest(it,:)];
    end
end

c2_classified_c0_D100 = [];
c2_classified_c1_D100 = [];
c2_classified_c2_D100 = [];
for it=1:1:size(c2_data_Dtest)
    p_x_0 = liklihood(c2_data_Dtest(it,:)', c0_gmm_D100.ComponentProportion', c0_gmm_D100.mu', c0_gmm_D100.Sigma);
    p_0 = p_x_0 * prior_D100(1);
    p_x_1 = liklihood(c2_data_Dtest(it,:)', c1_gmm_D100.ComponentProportion', c1_gmm_D100.mu', c1_gmm_D100.Sigma);
    p_1 = p_x_1 * prior_D100(2);
    p_x_2 = liklihood(c2_data_Dtest(it,:)', c2_gmm_D100.ComponentProportion', c2_gmm_D100.mu', c2_gmm_D100.Sigma);
    p_2 = p_x_2 * prior_D100(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c2_classified_c0_D100 = [c2_classified_c0_D100; c2_data_Dtest(it,:)];
    elseif class == 2
        c2_classified_c1_D100 = [c2_classified_c1_D100; c2_data_Dtest(it,:)];
    elseif class == 3
        c2_classified_c2_D100 = [c2_classified_c2_D100; c2_data_Dtest(it,:)];
    end
end

p_error_D100 = (size(c0_classified_c1_D100, 1) + size(c0_classified_c2_D100, 1) + size(c1_classified_c0_D100, 1) + size(c1_classified_c2_D100, 1) + size(c2_classified_c0_D100, 1) + size(c2_classified_c1_D100, 1)) / size(D10000_test,1);
%% ============================== Show D100 classifier results on Dtest ========================= %%
figure();
scatter(c0_classified_c0_D100(:,1), c0_classified_c0_D100(:,2), 'g*', 'DisplayName', 'c0 classified as c0'); hold on;
scatter(c0_classified_c1_D100(:,1), c0_classified_c1_D100(:,2), 'ro', 'DisplayName', 'c0 classified as c1'); hold on;
if ~isempty(c0_classified_c2_D100)
    scatter(c0_classified_c2_D100(:,1), c0_classified_c2_D100(:,2), 'r+', 'DisplayName', 'c0 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 0 data from Dtest using model trained with D100 train');
figure();
if ~isempty(c1_classified_c0_D100)
    scatter(c1_classified_c0_D100(:,1), c1_classified_c0_D100(:,2), 'r*', 'DisplayName', 'c1 classified as c0'); hold on;
end
if ~isempty(c1_classified_c1_D100)
    scatter(c1_classified_c1_D100(:,1), c1_classified_c1_D100(:,2), 'go', 'DisplayName', 'c1 classified as c1'); hold on;
end
if ~isempty(c1_classified_c2_D100)
    scatter(c1_classified_c2_D100(:,1), c1_classified_c2_D100(:,2), 'r+', 'DisplayName', 'c1 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 1 data from Dtest using model trained with D100 train');
figure();
if ~isempty(c2_classified_c0_D100)
    scatter(c2_classified_c0_D100(:,1), c2_classified_c0_D100(:,2), 'r*', 'DisplayName', 'c2 classified as c0'); hold on;
end
if ~isempty(c2_classified_c1_D100)
    scatter(c2_classified_c1_D100(:,1), c2_classified_c1_D100(:,2), 'ro', 'DisplayName', 'c2 classified as c1'); hold on;
end
if ~isempty(c2_classified_c2_D100)
    scatter(c2_classified_c2_D100(:,1), c2_classified_c2_D100(:,2), 'g+', 'DisplayName', 'c2 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 2 data from Dtest using model trained with D100 train');

%% ================================= MAP classifier from D500_train ============================= %%
prior_D500 = [size(c0_data_D500_train,1), size(c1_data_D500_train,1), size(c2_data_D500_train,1)] ./ size(D500_train, 1);

c0_classified_c0_D500 = [];
c0_classified_c1_D500 = [];
c0_classified_c2_D500 = [];

for it=1:1:size(c0_data_Dtest)
    p_x_0 = liklihood(c0_data_Dtest(it,:)', c0_gmm_D500.ComponentProportion', c0_gmm_D500.mu', c0_gmm_D500.Sigma);
    p_0 = p_x_0 * prior_D500(1);
    p_x_1 = liklihood(c0_data_Dtest(it,:)', c1_gmm_D500.ComponentProportion', c1_gmm_D500.mu', c1_gmm_D500.Sigma);
    p_1 = p_x_1 * prior_D500(2);
    p_x_2 = liklihood(c0_data_Dtest(it,:)', c2_gmm_D500.ComponentProportion', c2_gmm_D500.mu', c2_gmm_D500.Sigma);
    p_2 = p_x_2 * prior_D500(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c0_classified_c0_D500 = [c0_classified_c0_D500; c0_data_Dtest(it,:)];
    elseif class == 2
        c0_classified_c1_D500 = [c0_classified_c1_D500; c0_data_Dtest(it,:)];
    elseif class == 3
        c0_classified_c2_D500 = [c0_classified_c2_D500; c0_data_Dtest(it,:)];
    end
end

c1_classified_c0_D500 = [];
c1_classified_c1_D500 = [];
c1_classified_c2_D500 = [];
for it=1:1:size(c1_data_Dtest)
    p_x_0 = liklihood(c1_data_Dtest(it,:)', c0_gmm_D500.ComponentProportion', c0_gmm_D500.mu', c0_gmm_D500.Sigma);
    p_0 = p_x_0 * prior_D500(1);
    p_x_1 = liklihood(c1_data_Dtest(it,:)', c1_gmm_D500.ComponentProportion', c1_gmm_D500.mu', c1_gmm_D500.Sigma);
    p_1 = p_x_1 * prior_D500(2);
    p_x_2 = liklihood(c1_data_Dtest(it,:)', c2_gmm_D500.ComponentProportion', c2_gmm_D500.mu', c2_gmm_D500.Sigma);
    p_2 = p_x_2 * prior_D500(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c1_classified_c0_D500 = [c1_classified_c0_D500; c1_data_Dtest(it,:)];
    elseif class == 2
        c1_classified_c1_D500 = [c1_classified_c1_D500; c1_data_Dtest(it,:)];
    elseif class == 3
        c1_classified_c2_D500 = [c1_classified_c2_D500; c1_data_Dtest(it,:)];
    end
end

c2_classified_c0_D500 = [];
c2_classified_c1_D500 = [];
c2_classified_c2_D500 = [];
for it=1:1:size(c2_data_Dtest)
    p_x_0 = liklihood(c2_data_Dtest(it,:)', c0_gmm_D500.ComponentProportion', c0_gmm_D500.mu', c0_gmm_D500.Sigma);
    p_0 = p_x_0 * prior_D500(1);
    p_x_1 = liklihood(c2_data_Dtest(it,:)', c1_gmm_D500.ComponentProportion', c1_gmm_D500.mu', c1_gmm_D500.Sigma);
    p_1 = p_x_1 * prior_D500(2);
    p_x_2 = liklihood(c2_data_Dtest(it,:)', c2_gmm_D500.ComponentProportion', c2_gmm_D500.mu', c2_gmm_D500.Sigma);
    p_2 = p_x_2 * prior_D500(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c2_classified_c0_D500 = [c2_classified_c0_D500; c2_data_Dtest(it,:)];
    elseif class == 2
        c2_classified_c1_D500 = [c2_classified_c1_D500; c2_data_Dtest(it,:)];
    elseif class == 3
        c2_classified_c2_D500 = [c2_classified_c2_D500; c2_data_Dtest(it,:)];
    end
end

p_error_D500 = (size(c0_classified_c1_D500, 1) + size(c0_classified_c2_D500, 1) + size(c1_classified_c0_D500, 1) + size(c1_classified_c2_D500, 1) + size(c2_classified_c0_D500, 1) + size(c2_classified_c1_D500, 1)) / size(D10000_test,1);
%% ============================== Show D500 classifier results on Dtest ========================= %%
figure();
scatter(c0_classified_c0_D500(:,1), c0_classified_c0_D500(:,2), 'g*', 'DisplayName', 'c0 classified as c0'); hold on;
scatter(c0_classified_c1_D500(:,1), c0_classified_c1_D500(:,2), 'ro', 'DisplayName', 'c0 classified as c1'); hold on;
if ~isempty(c0_classified_c2_D500)
    scatter(c0_classified_c2_D500(:,1), c0_classified_c2_D500(:,2), 'r+', 'DisplayName', 'c0 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 0 data from Dtest using model trained with D500 train');
figure();
if ~isempty(c1_classified_c0_D500)
    scatter(c1_classified_c0_D500(:,1), c1_classified_c0_D500(:,2), 'r*', 'DisplayName', 'c1 classified as c0'); hold on;
end
if ~isempty(c1_classified_c1_D500)
    scatter(c1_classified_c1_D500(:,1), c1_classified_c1_D500(:,2), 'go', 'DisplayName', 'c1 classified as c1'); hold on;
end
if ~isempty(c1_classified_c2_D500)
    scatter(c1_classified_c2_D500(:,1), c1_classified_c2_D500(:,2), 'r+', 'DisplayName', 'c1 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 1 data from Dtest using model trained with D500 train');
figure();
if ~isempty(c2_classified_c0_D500)
    scatter(c2_classified_c0_D500(:,1), c2_classified_c0_D500(:,2), 'r*', 'DisplayName', 'c2 classified as c0'); hold on;
end
if ~isempty(c2_classified_c1_D500)
    scatter(c2_classified_c1_D500(:,1), c2_classified_c1_D500(:,2), 'ro', 'DisplayName', 'c2 classified as c1'); hold on;
end
if ~isempty(c2_classified_c2_D500)
    scatter(c2_classified_c2_D500(:,1), c2_classified_c2_D500(:,2), 'g+', 'DisplayName', 'c2 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 2 data from Dtest using model trained with D500 train');


%% ================================= MAP classifier from D1000_train ============================= %%
prior_D1000 = [size(c0_data_D1000_train,1), size(c1_data_D1000_train,1), size(c2_data_D1000_train,1)] ./ size(D1000_train, 1);

c0_classified_c0_D1000 = [];
c0_classified_c1_D1000 = [];
c0_classified_c2_D1000 = [];

for it=1:1:size(c0_data_Dtest)
    p_x_0 = liklihood(c0_data_Dtest(it,:)', c0_gmm_D1000.ComponentProportion', c0_gmm_D1000.mu', c0_gmm_D1000.Sigma);
    p_0 = p_x_0 * prior_D1000(1);
    p_x_1 = liklihood(c0_data_Dtest(it,:)', c1_gmm_D1000.ComponentProportion', c1_gmm_D1000.mu', c1_gmm_D1000.Sigma);
    p_1 = p_x_1 * prior_D1000(2);
    p_x_2 = liklihood(c0_data_Dtest(it,:)', c2_gmm_D1000.ComponentProportion', c2_gmm_D1000.mu', c2_gmm_D1000.Sigma);
    p_2 = p_x_2 * prior_D1000(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c0_classified_c0_D1000 = [c0_classified_c0_D1000; c0_data_Dtest(it,:)];
    elseif class == 2
        c0_classified_c1_D1000 = [c0_classified_c1_D1000; c0_data_Dtest(it,:)];
    elseif class == 3
        c0_classified_c2_D1000 = [c0_classified_c2_D1000; c0_data_Dtest(it,:)];
    end
end

c1_classified_c0_D1000 = [];
c1_classified_c1_D1000 = [];
c1_classified_c2_D1000 = [];
for it=1:1:size(c1_data_Dtest)
    p_x_0 = liklihood(c1_data_Dtest(it,:)', c0_gmm_D1000.ComponentProportion', c0_gmm_D1000.mu', c0_gmm_D1000.Sigma);
    p_0 = p_x_0 * prior_D1000(1);
    p_x_1 = liklihood(c1_data_Dtest(it,:)', c1_gmm_D1000.ComponentProportion', c1_gmm_D1000.mu', c1_gmm_D1000.Sigma);
    p_1 = p_x_1 * prior_D1000(2);
    p_x_2 = liklihood(c1_data_Dtest(it,:)', c2_gmm_D1000.ComponentProportion', c2_gmm_D1000.mu', c2_gmm_D1000.Sigma);
    p_2 = p_x_2 * prior_D1000(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c1_classified_c0_D1000 = [c1_classified_c0_D1000; c1_data_Dtest(it,:)];
    elseif class == 2
        c1_classified_c1_D1000 = [c1_classified_c1_D1000; c1_data_Dtest(it,:)];
    elseif class == 3
        c1_classified_c2_D1000 = [c1_classified_c2_D1000; c1_data_Dtest(it,:)];
    end
end

c2_classified_c0_D1000 = [];
c2_classified_c1_D1000 = [];
c2_classified_c2_D1000 = [];
for it=1:1:size(c2_data_Dtest)
    p_x_0 = liklihood(c2_data_Dtest(it,:)', c0_gmm_D1000.ComponentProportion', c0_gmm_D1000.mu', c0_gmm_D1000.Sigma);
    p_0 = p_x_0 * prior_D1000(1);
    p_x_1 = liklihood(c2_data_Dtest(it,:)', c1_gmm_D1000.ComponentProportion', c1_gmm_D1000.mu', c1_gmm_D1000.Sigma);
    p_1 = p_x_1 * prior_D1000(2);
    p_x_2 = liklihood(c2_data_Dtest(it,:)', c2_gmm_D1000.ComponentProportion', c2_gmm_D1000.mu', c2_gmm_D1000.Sigma);
    p_2 = p_x_2 * prior_D1000(3);
    [p, class] = max([p_0, p_1, p_2]);
    
    if class == 1
        c2_classified_c0_D1000 = [c2_classified_c0_D1000; c2_data_Dtest(it,:)];
    elseif class == 2
        c2_classified_c1_D1000 = [c2_classified_c1_D1000; c2_data_Dtest(it,:)];
    elseif class == 3
        c2_classified_c2_D1000 = [c2_classified_c2_D1000; c2_data_Dtest(it,:)];
    end
end

p_error_D1000 = (size(c0_classified_c1_D1000, 1) + size(c0_classified_c2_D1000, 1) + size(c1_classified_c0_D1000, 1) + size(c1_classified_c2_D1000, 1) + size(c2_classified_c0_D1000, 1) + size(c2_classified_c1_D1000, 1)) / size(D10000_test,1);

%% ============================== Show D1000 classifier results on Dtest ========================= %%
figure();
scatter(c0_classified_c0_D1000(:,1), c0_classified_c0_D1000(:,2), 'g*', 'DisplayName', 'c0 classified as c0'); hold on;
scatter(c0_classified_c1_D1000(:,1), c0_classified_c1_D1000(:,2), 'ro', 'DisplayName', 'c0 classified as c1'); hold on;
if ~isempty(c0_classified_c2_D1000)
    scatter(c0_classified_c2_D1000(:,1), c0_classified_c2_D1000(:,2), 'r+', 'DisplayName', 'c0 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 0 data from Dtest using model trained with D1000 train');
figure();
if ~isempty(c1_classified_c0_D1000)
    scatter(c1_classified_c0_D1000(:,1), c1_classified_c0_D1000(:,2), 'r*', 'DisplayName', 'c1 classified as c0'); hold on;
end
if ~isempty(c1_classified_c1_D1000)
    scatter(c1_classified_c1_D1000(:,1), c1_classified_c1_D1000(:,2), 'go', 'DisplayName', 'c1 classified as c1'); hold on;
end
if ~isempty(c1_classified_c2_D1000)
    scatter(c1_classified_c2_D1000(:,1), c1_classified_c2_D1000(:,2), 'r+', 'DisplayName', 'c1 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 1 data from Dtest using model trained with D1000 train');
figure();
if ~isempty(c2_classified_c0_D1000)
    scatter(c2_classified_c0_D1000(:,1), c2_classified_c0_D1000(:,2), 'r*', 'DisplayName', 'c2 classified as c0'); hold on;
end
if ~isempty(c2_classified_c1_D1000)
    scatter(c2_classified_c1_D1000(:,1), c2_classified_c1_D1000(:,2), 'ro', 'DisplayName', 'c2 classified as c1'); hold on;
end
if ~isempty(c2_classified_c2_D1000)
    scatter(c2_classified_c2_D1000(:,1), c2_classified_c2_D1000(:,2), 'g+', 'DisplayName', 'c2 classified as c2'); hold on;
end
legend();
xlabel('x1');
ylabel('x2');
title('Classification of Class 2 data from Dtest using model trained with D1000 train');

%% ===================== All Functions reside here =========================== %%
function p_x = liklihood(x, alpha, mu, Sigma)
    N = size(x, 2);  
    M = size(alpha,1);
    d = size(mu, 1);
%     fprintf("\nEvaluating GMM of order %d with %d samples..\n", M, N);
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    p_x = sum(temp, 1);
end

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