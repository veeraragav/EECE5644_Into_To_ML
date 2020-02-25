clc
clear

%% ============ Sample Generation (Demo) ============ %%
N = 10; % no of samples
h = @(x, w) (w(1) * x.^3) + (w(2) * x.^2) + (w(3) * x) + w(4);
w_true = [1; -0.2; -0.29; 0.03];
sigma_v = 0.1;
% gamma = 1; %remove later
% ========= Display generated data points (Demo)========== %%
x = -1 + 2*rand(N, 1); %sampled from U[-1,1]
sigma_v = 0.5; %additive noise variance
y = h(x, w_true) + normrnd(0, sigma_v, [N 1]);
z = [ (x.^3)'; (x.^2)'; x'; ones(1, N)]; 
figure(1);
grid on; hold on;
scatter(x', y');
hold on;
fplot(@(X) h(X, w_true), [-1.5 1.5]);
xlabel('x'), ylabel('y');
legend('Data Samples', 'Cubic Polynomial Model');
title('Data Samples and Corresponding Model');

%% ============ ML estimation  ================ %%
x = -1 + 2*rand(N, 1); %sampled from U[-1,1]
y = h(x, w_true) + normrnd(0, sigma_v, [N 1]);
z = [ (x.^3)'; (x.^2)'; x'; ones(1, N)]; 
zzT = [];
for i = 1: N
    zzT(:, :, i) = z(:, i) * z(:, i)';
end
w_est_ml = ((sum(zzT, 3) ^-1) *(sum(repmat(y',4,1).*z, 2)));

%% ============ MAP estimator function ================ %%
zzT = [];
for i = 1: N
    zzT(:, :, i) = z(:, i) * z(:, i)';
end
w_est_map = @(gamma) (( sum(zzT, 3) + (sigma_v/gamma)*eye(size(z, 1)))^-1) *(sum( repmat(y',4,1).*z, 2));

%% =============== MAP estimation over a range of gamma values =========== %%
gamma_array = 10.^[-10:0.01:3];
w_est_map_array = [];
for i = 1 : size(gamma_array,2)
    w_est_map_array(:, i) = w_est_map(gamma_array(i));
end

%% ========== Plot results from above experiments ============== %%
figure(2);
plot(gamma_array, w_est_map_array(1, :), 'r-' , 'DisplayName', 'a(MAP)'); hold on;
plot(gamma_array, w_est_map_array(2, :), 'g-', 'DisplayName', 'b(MAP)'); hold on;
plot(gamma_array, w_est_map_array(3, :), 'b-', 'DisplayName', 'c(MAP)'); hold on;
plot(gamma_array, w_est_map_array(4, :), 'm-', 'DisplayName', 'd(MAP)'); hold on;

plot(gamma_array, repmat(w_est_ml(1), 1, size(gamma_array, 2)), 'r--', 'DisplayName', 'a(ML)'); hold on;
plot(gamma_array, repmat(w_est_ml(2), 1, size(gamma_array, 2)), 'g--', 'DisplayName', 'b(ML)'); hold on;
plot(gamma_array, repmat(w_est_ml(3), 1, size(gamma_array, 2)), 'b--', 'DisplayName', 'c(ML)'); hold on;
plot(gamma_array, repmat(w_est_ml(4), 1, size(gamma_array, 2)), 'm--', 'DisplayName', 'd(ML)'); hold on;
legend();
xlabel('gamma'), ylabel('estimates');

%% ======================== Squared L2 norm function ======================= %%
normm = @(w) (w(1) - w_true(1))^2 + (w(2) - w_true(2))^2 + (w(3) - w_true(3))^2 + (w(4) - w_true(4))^2;
% nnorm = @(w) (w(1) - w_est_ml(1))^2 + (w(2) - w_est_ml(2))^2 + (w(3) - w_est_ml(3))^2 + (w(4) - w_est_ml(4))^2;
%% =========== Calculating Squared L2 Norm for 100 experiments for a range of gamma  ======== %%
% percentiles_25 =[];
% percentiles_50 =[];
% percentiles_75 =[];
% gamma_array = 10.^[-10 : 0.01 : 2];
% for i = 1 : size(gamma_array,2)
%     norm_array = [];
%     
%     for j = 1 : 100
%         x = [];
%         y = [];
%         z = [];
%         zzT = [];
%         x = -1 + 2*rand(N, 1); 
%         z = [ (x.^3)'; (x.^2)'; x'; ones(1, N)]; 
%         for k = 1: N
%              zzT(:, :, k) = z(:, k) * z(:, k)';
%         end
%         w_est = (( sum(zzT, 3) + (sigma_v/gamma_array(i))*eye(size(z, 1)))^-1) *(sum( repmat(y',4,1).*z, 2));
%         norm_array = [norm_array, norm(w_est)];
%     end
%     percentiles_25 = [percentiles_25 prctile(norm_array, 25)];
%     percentiles_50 = [percentiles_50 prctile(norm_array, 50)];
%     percentiles_75 = [percentiles_75 prctile(norm_array, 75)];
% end
%% ===== Calculating Squared L2 Norm for 100 experiments for a range of gamma ===== %%
percentiles_25 =[];
percentiles_50 =[];
percentiles_75 =[];
minn = [];
maxx = [];
gamma_array = 10.^[-5 : 0.005 : 5];
norm_array = [];
ml_morm = [];
for j = 1 : 100
    x = [];
    y = [];
    z = [];
    zzT = [];
    x = -1 + 2*rand(N, 1); 
    y = h(x, w_true) + normrnd(0, sigma_v, [N 1]);
    z = [ (x.^3)'; (x.^2)'; x'; ones(1, N)]; 
    for k = 1: N
         zzT(:, :, k) = z(:, k) * z(:, k)';
    end
    w_est_ml = ((sum(zzT, 3) ^-1) *(sum(repmat(y',4,1).*z, 2)));
    ml_norm(j) = normm(w_est_ml);
    for i = 1 : size(gamma_array,2)
        w_est = (( sum(zzT, 3) + (sigma_v/gamma_array(i))*eye(size(z, 1)))^-1) *(sum( repmat(y',4,1).*z, 2));
        norm_array(j, i) = normm(w_est);
    end
end

%% ==== Calculate Percentiles ==== %%
for k = 1:1:size(gamma_array,2)
    percentiles_25(k) = prctile(norm_array(:, k), 25);
    percentiles_50(k) = prctile(norm_array(:, k), 50);
    percentiles_75(k) = prctile(norm_array(:, k), 75);
    minn(k) = min(norm_array(:, k));
    maxx(k) = max(norm_array(:, k));
end
percentiles_25_ml = prctile(ml_norm, 25);
percentiles_50_ml = prctile(ml_norm, 50);
percentiles_75_ml = prctile(ml_norm, 75);
% %% === trial plot === %%
% figure(10);
% for k = 1:1:100
%     hold on;
%     semilogx([-4:0.01:3], norm_array(k, :));
% end
% figure(11);
% for k = 1:1:100
%     hold on;
%     semilogx([-4:0.01:3], nnorm_array(k, :));
% end
%% ========== Plot results 1 from above experiment ============== %%
figure(3);
axis equal;
plot(gamma_array, percentiles_25, 'r-'); hold on;
plot(gamma_array, percentiles_50, 'g-'); hold on;
plot(gamma_array, percentiles_75, 'b-'); hold on;
plot(gamma_array, minn); hold on;
plot(gamma_array, maxx); hold on;
set(gca,'Xscale','log');
legend('25th percentile', '50th percentile', '75th percentile', 'minimum', 'maximum');
xlabel('gamma^2'), ylabel('squared L2 Norm');
title('Squared L2 norm of MAP estimates for a range of gamma^2');
%% ========== Plot results 2 from above experiment ============== %%
figure(4);
axis equal;
plot(gamma_array, repmat(percentiles_25_ml, 1, size(gamma_array, 2)), 'r--'); hold on;
plot(gamma_array, repmat(percentiles_50_ml, 1, size(gamma_array, 2)), 'g--'); hold on;
plot(gamma_array, repmat(percentiles_75_ml, 1, size(gamma_array, 2)), 'b--'); hold on;
plot(gamma_array, percentiles_25, 'r-'); hold on;
plot(gamma_array, percentiles_50, 'g-'); hold on;
plot(gamma_array, percentiles_75, 'b-'); hold on;
set(gca,'Xscale','log');
legend('25th percentile(ML)', '50th percentile(ML)', '75th percentile(ML)','25th percentile(MAP)', '50th percentile(MAP)', '75th percentile(MAP)');
xlabel('gamma^2'), ylabel('squared L2 Norm');
title('Squared L2 norm of MAP and ML estimates');