clc
clear

%% ================== Generating D_10_train =============== %%
n = 2; % number of feature dimensions
N = 10; % number of iid samples
mu(:,1) = [-2; 0];
mu(:,2) = [2; 0];
Sigma(:,:,1) = [ 1 -0.9; -0.9, 2 ]; 
Sigma(:,:,2) = [ 2 0.9; 0.9, 1 ]; 
prior = [0.9,0.1]; % class priors for labels 0 and 1 respectively
label_D_10_train = rand(1,N) >= prior(1);
Nc = [length(find(label_D_10_train==0)),length(find(label_D_10_train==1))]; % number of samples from each class
D_10_train = zeros(n,N); 
% Draw samples from each class pdf
for l = 0:1
    D_10_train(:,label_D_10_train==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
c0_data = [];
c1_data = [];
for i = 1 : N
    if label_D_10_train(i) == 0
        c0_data = [c0_data; D_10_train(:, i)'];
    end
    if label_D_10_train(i) == 1
        c1_data = [c1_data; D_10_train(:, i)'];
    end
end

figure(1), clf,
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
if ~ isempty(c1_data)
    plot(c1_data(:,1),c1_data(:,2),'+')
end
axis equal,
legend('Class 0','Class 1'), 
title('D 10 train'),
xlabel('x_1'), ylabel('x_2'),

%% ================== Generating D_100_train =============== %%
N = 100;
label_D_100_train = rand(1,N) >= prior(1);
Nc = [length(find(label_D_100_train==0)),length(find(label_D_100_train==1))]; % number of samples from each class
D_100_train = zeros(n,N); 
% Draw samples from each class pdf
for l = 0:1
    D_100_train(:,label_D_100_train==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
c0_data = [];
c1_data = [];
for i = 1 : N
    if label_D_100_train(i) == 0
        c0_data = [c0_data; D_100_train(:, i)'];
    end
    if label_D_100_train(i) == 1
        c1_data = [c1_data; D_100_train(:, i)'];
    end
end

figure(2), clf,
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
legend('Class 0','Class 1'), 
title('D 100 train'),
xlabel('x_1'), ylabel('x_2'),

%% ================== Generating D_1000_train =============== %%
N = 1000;
label_D_1000_train = rand(1,N) >= prior(1);
Nc = [length(find(label_D_1000_train==0)),length(find(label_D_1000_train==1))]; % number of samples from each class
D_1000_train = zeros(n,N); 
% Draw samples from each class pdf
for l = 0:1
    D_1000_train(:,label_D_1000_train==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
c0_data = [];
c1_data = [];
for i = 1 : N
    if label_D_1000_train(i) == 0
        c0_data = [c0_data; D_1000_train(:, i)'];
    end
    if label_D_1000_train(i) == 1
        c1_data = [c1_data; D_1000_train(:, i)'];
    end
end

figure(3), clf,
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
legend('Class 0','Class 1'), 
title('D 1000 train'),
xlabel('x_1'), ylabel('x_2'),

%% ================== Generating D_10000_validate =============== %%
N = 10000;
label_D_10000_validate = rand(1,N) >= prior(1);
Nc = [length(find(label_D_10000_validate==0)),length(find(label_D_10000_validate==1))]; % number of samples from each class
D_10000_validate = zeros(n,N); 
% Draw samples from each class pdf
for l = 0:1
    D_10000_validate(:,label_D_10000_validate==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
c0_data = [];
c1_data = [];
for i = 1 : N
    if label_D_10000_validate(i) == 0
        c0_data = [c0_data; D_10000_validate(:, i)'];
    end
    if label_D_10000_validate(i) == 1
        c1_data = [c1_data; D_10000_validate(:, i)'];
    end
end

figure(4), clf,
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
legend('Class 0','Class 1'), 
title('D 10000 validate'),
xlabel('x_1'), ylabel('x_2'),

%% =============== Saving all datasets =================== %%
save('D_10_train.mat', 'D_10_train', 'label_D_10_train');
save('D_100_train.mat', 'D_100_train', 'label_D_100_train');
save('D_1000_train.mat', 'D_1000_train', 'label_D_1000_train');
save('D_10000_validate.mat', 'D_10000_validate', 'label_D_10000_validate');