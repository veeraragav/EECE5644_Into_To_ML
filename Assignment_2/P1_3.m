clc
clear
%% ========================= Loading Datasets ======================= %%
% This is generated by 'sample_generation_P1.m'
% Please run the above script first before running this script
load('D_10_train.mat');
load('D_100_train.mat');
load('D_1000_train.mat');
load('D_10000_validate.mat');
n = 2;
warning('off','all');
%% ======================== Logistic Regression over D_10_train ======================= %%
Nc_D_10_train = [length(find(label_D_10_train==0)),length(find(label_D_10_train==1))];
N_D_10_train = sum(Nc_D_10_train);
z_D_10_train = [ones(1, N_D_10_train); D_10_train; D_10_train(1,:).^2; D_10_train(1,:) .* D_10_train(2,:);  D_10_train(2,:).^2];
initial_theta_D_10_train = [0.1; 0.1; 0.1; 0.1; 0.1; 0.1 ];
options = optimset('MaxFunEvals', 6000);
[theta_D_10_train, cost_D_10_train] = fminsearch(@(t)(cost_func(t, z_D_10_train, label_D_10_train, N_D_10_train)), initial_theta_D_10_train);
decision_boundary_D_10_train = @(x1,x2) theta_D_10_train(1) + theta_D_10_train(2)*x1 + theta_D_10_train(3)*x2 + theta_D_10_train(4)*x1*x1 + theta_D_10_train(5)*x1*x2 + theta_D_10_train(6)*x2*x2;


%% =========================== D_10000_validate dataset is divided into different lists based on class label  ======================= %%
c0_data = [];
c1_data = [];
for i = 1 : size(D_10000_validate,2)
    if label_D_10000_validate(i) == 0
        c0_data = [c0_data; D_10000_validate(:, i)'];
    end
    if label_D_10000_validate(i) == 1
        c1_data = [c1_data; D_10000_validate(:, i)'];
    end
end

%% ============= Classifying D_10000_validate using Model trained with D_10_train =========== %%
c0_classified_as_c0 = [];
c0_classified_as_c1 = [];

for i = 1 : 1 : size(c0_data,1)
    label = decision_boundary_D_10_train(c0_data(i, 1), c0_data(i, 2)) >= 0;

    if label == 0
        c0_classified_as_c0 = [c0_classified_as_c0; (c0_data(i, :)) ];
    end

    if label == 1
        c0_classified_as_c1 = [c0_classified_as_c1; (c0_data(i, :)) ];
    end

end

%classifying class 1 dataset
c1_classified_as_c0 = [];
c1_classified_as_c1 = [];
for i = 1 : 1 : size(c1_data,1)
    label = decision_boundary_D_10_train(c1_data(i, 1), c1_data(i, 2)) > 0;

    if label == 0
        c1_classified_as_c0 = [c1_classified_as_c0; (c1_data(i, :)) ];
    end

    if label == 1
        c1_classified_as_c1 = [c1_classified_as_c1; (c1_data(i, :)) ];
    end

end

p_error_D_10_train = (size(c1_classified_as_c0,1) + size(c0_classified_as_c1,1)) / size(D_10000_validate,2);

%% ========== Plotting validation data after classification using Model trained with D_10_train ======= %%
figure(1);
plot(c0_classified_as_c0(:,1),c0_classified_as_c0(:,2), 'og', 'DisplayName', 'Class0 classified as Class0');
hold on;
plot(c1_classified_as_c1(:,1), c1_classified_as_c1(:,2), '+g', 'DisplayName', 'Class1 classified as Class1');

if ~ isempty(c0_classified_as_c1)
    hold on;
    plot(c0_classified_as_c1(:,1),c0_classified_as_c1(:,2), 'or', 'DisplayName', 'Class0 classified as Class1');
end

if ~ isempty(c1_classified_as_c0)
    hold on;
    plot(c1_classified_as_c0(:,1),c1_classified_as_c0(:,2), '+r', 'DisplayName', 'Class1 classified as Class0');
end

hold on;
fimplicit(decision_boundary_D_10_train, [-10, 10], 'DisplayName', 'Decision boundary');
xlabel('x_1'), ylabel('x_2');
title('Classification using logistic quadratic function model trained with D10train');
legend();
axis equal;
hold off;

%% =========================== D_10_train dataset plot  ======================= %%
c0_data = [];
c1_data = [];
for i = 1 : size(D_10_train,2)
    if label_D_10_train(i) == 0
        c0_data = [c0_data; D_10_train(:, i)'];
    end
    if label_D_10_train(i) == 1
        c1_data = [c1_data; D_10_train(:, i)'];
    end
end

figure(2);
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
hold on;
fimplicit(decision_boundary_D_10_train, [-6, 6]);
legend('Class 0','Class 1', 'Decision boundary'), 
title('D10train and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
axis equal;
%% ========================= End of D_10_train model analysis ============================ %%




%% ======================== Logistic Regression over D_100_train ======================= %%
Nc_D_100_train = [length(find(label_D_100_train==0)),length(find(label_D_100_train==1))];
N_D_100_train = sum(Nc_D_100_train);
z_D_100_train = [ones(1, N_D_100_train); D_100_train; D_100_train(1,:).^2; D_100_train(1,:) .* D_100_train(2,:);  D_100_train(2,:).^2];
initial_theta_D_100_train = [-1.7771; 3.7698; -0.5685; 0; 0; 0 ];
options = optimset('MaxFunEvals', 6000);
[theta_D_100_train, cost_D_100_train] = fminsearch(@(t)(cost_func(t, z_D_100_train, label_D_100_train, N_D_100_train)), initial_theta_D_100_train, options);
decision_boundary_D_100_train = @(x1,x2) theta_D_100_train(1) + theta_D_100_train(2)*x1 + theta_D_100_train(3)*x2 + theta_D_100_train(4)*x1*x1 + theta_D_100_train(5)*x1*x2 + theta_D_100_train(6)*x2*x2;


%% ============= D_10000_validate dataset is divided into different lists based on class label  =============== %%
c0_data = [];
c1_data = [];
for i = 1 : size(D_10000_validate,2)
    if label_D_10000_validate(i) == 0
        c0_data = [c0_data; D_10000_validate(:, i)'];
    end
    if label_D_10000_validate(i) == 1
        c1_data = [c1_data; D_10000_validate(:, i)'];
    end
end

%% ============= Classifying D_10000_validate using Model trained with D_10_train =========== %%
c0_classified_as_c0 = [];
c0_classified_as_c1 = [];

for i = 1 : 1 : size(c0_data,1)
    label = decision_boundary_D_100_train(c0_data(i, 1), c0_data(i, 2)) >= 0;

    if label == 0
        c0_classified_as_c0 = [c0_classified_as_c0; (c0_data(i, :)) ];
    end

    if label == 1
        c0_classified_as_c1 = [c0_classified_as_c1; (c0_data(i, :)) ];
    end

end

%classifying class 1 dataset
c1_classified_as_c0 = [];
c1_classified_as_c1 = [];
for i = 1 : 1 : size(c1_data,1)
    label = decision_boundary_D_100_train(c1_data(i, 1), c1_data(i, 2)) > 0;

    if label == 0
        c1_classified_as_c0 = [c1_classified_as_c0; (c1_data(i, :)) ];
    end

    if label == 1
        c1_classified_as_c1 = [c1_classified_as_c1; (c1_data(i, :)) ];
    end

end

p_error_D_100_train = (size(c1_classified_as_c0,1) + size(c0_classified_as_c1,1)) / size(D_10000_validate,2);

%% ========== Plotting validation data after classification using Model trained with D_100_train ======= %%
figure(3);
plot(c0_classified_as_c0(:,1),c0_classified_as_c0(:,2), 'og', 'DisplayName', 'Class0 classified as Class0');
hold on;
plot(c1_classified_as_c1(:,1), c1_classified_as_c1(:,2), '+g', 'DisplayName', 'Class1 classified as Class1');

if ~ isempty(c0_classified_as_c1)
    hold on;
    plot(c0_classified_as_c1(:,1),c0_classified_as_c1(:,2), 'or', 'DisplayName', 'Class0 classified as Class1');
end

if ~ isempty(c1_classified_as_c0)
    hold on;
    plot(c1_classified_as_c0(:,1),c1_classified_as_c0(:,2), '+r', 'DisplayName', 'Class1 classified as Class0');
end

hold on;
fimplicit(decision_boundary_D_100_train, [-10, 10], 'DisplayName', 'Decision boundary');
xlabel('x_1'), ylabel('x_2');
title('Classification using logistic quadratic function model trained with D100train');
legend();
axis equal;
hold off;


%% =========================== D_100_train dataset plot  ======================= %%
c0_data = [];
c1_data = [];
for i = 1 : size(D_100_train,2)
    if label_D_100_train(i) == 0
        c0_data = [c0_data; D_100_train(:, i)'];
    end
    if label_D_100_train(i) == 1
        c1_data = [c1_data; D_100_train(:, i)'];
    end
end

figure(4);
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
hold on;
fimplicit(decision_boundary_D_100_train, [-6, 6]);
legend('Class 0','Class 1', 'Decision boundary'), 
title('D100train and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
axis equal;
%% ========================= End of D_100_train model analysis ============================ %%


%% ======================== Logistic Regression over D_1000_train ======================= %%
Nc_D_1000_train = [length(find(label_D_1000_train==0)),length(find(label_D_1000_train==1))];
N_D_1000_train = sum(Nc_D_1000_train);
z_D_1000_train = [ones(1, N_D_1000_train); D_1000_train; D_1000_train(1,:).^2; D_1000_train(1,:) .* D_1000_train(2,:);  D_1000_train(2,:).^2];
initial_theta_D_1000_train = [-1.0754; 5.4024; 1.1070; 0; 0; 0 ];
options = optimset('MaxFunEvals', 6000);
[theta_D_1000_train, cost_D_1000_train] = fminsearch(@(t)(cost_func(t, z_D_1000_train, label_D_1000_train, N_D_1000_train)), initial_theta_D_1000_train, options);
decision_boundary_D_1000_train = @(x1,x2) theta_D_1000_train(1) + theta_D_1000_train(2)*x1 + theta_D_1000_train(3)*x2 + theta_D_1000_train(4)*x1*x1 + theta_D_1000_train(5)*x1*x2 + theta_D_1000_train(6)*x2*x2;



%% ============= D_10000_validate dataset is divided into different lists based on class label  =============== %%
c0_data = [];
c1_data = [];
for i = 1 : size(D_10000_validate,2)
    if label_D_10000_validate(i) == 0
        c0_data = [c0_data; D_10000_validate(:, i)'];
    end
    if label_D_10000_validate(i) == 1
        c1_data = [c1_data; D_10000_validate(:, i)'];
    end
end

%% ============= Classifying D_10000_validate using Model trained with D_10_train =========== %%
c0_classified_as_c0 = [];
c0_classified_as_c1 = [];

for i = 1 : 1 : size(c0_data,1)
    label = decision_boundary_D_1000_train(c0_data(i, 1), c0_data(i, 2)) >= 0;

    if label == 0
        c0_classified_as_c0 = [c0_classified_as_c0; (c0_data(i, :)) ];
    end

    if label == 1
        c0_classified_as_c1 = [c0_classified_as_c1; (c0_data(i, :)) ];
    end

end

%classifying class 1 dataset
c1_classified_as_c0 = [];
c1_classified_as_c1 = [];
for i = 1 : 1 : size(c1_data,1)
    label = decision_boundary_D_1000_train(c1_data(i, 1), c1_data(i, 2)) > 0;

    if label == 0
        c1_classified_as_c0 = [c1_classified_as_c0; (c1_data(i, :)) ];
    end

    if label == 1
        c1_classified_as_c1 = [c1_classified_as_c1; (c1_data(i, :)) ];
    end

end

p_error_D_1000_train = (size(c1_classified_as_c0,1) + size(c0_classified_as_c1,1)) / size(D_10000_validate,2);

%% ========== Plotting validation data after classification using Model trained with D_100_train ======= %%
figure(5);
plot(c0_classified_as_c0(:,1),c0_classified_as_c0(:,2), 'og', 'DisplayName', 'Class0 classified as Class0');
hold on;
plot(c1_classified_as_c1(:,1), c1_classified_as_c1(:,2), '+g', 'DisplayName', 'Class1 classified as Class1');

if ~ isempty(c0_classified_as_c1)
    hold on;
    plot(c0_classified_as_c1(:,1),c0_classified_as_c1(:,2), 'or', 'DisplayName', 'Class0 classified as Class1');
end

if ~ isempty(c1_classified_as_c0)
    hold on;
    plot(c1_classified_as_c0(:,1),c1_classified_as_c0(:,2), '+r', 'DisplayName', 'Class1 classified as Class0');
end

hold on;
fimplicit(decision_boundary_D_1000_train, [-10, 10], 'DisplayName', 'Decision boundary');
xlabel('x_1'), ylabel('x_2');
title('Classification using logistic quadratic function model trained with D1000train');
legend();
axis equal;
hold off;


%% =========================== D_1000_train dataset plot  ======================= %%
c0_data = [];
c1_data = [];
for i = 1 : size(D_1000_train,2)
    if label_D_1000_train(i) == 0
        c0_data = [c0_data; D_1000_train(:, i)'];
    end
    if label_D_1000_train(i) == 1
        c1_data = [c1_data; D_1000_train(:, i)'];
    end
end

figure(6);
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
hold on;
fimplicit(decision_boundary_D_1000_train, [-6, 6]);
legend('Class 0','Class 1', 'Decision boundary'), 
title('D1000train and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
axis equal;
%% ========================= End of D_1000_train model analysis ============================ %%

%% ================ Plot P(error) for all 3 models ================== %%
figure(7);
X = categorical({'D10train', 'D100train', 'D1000train'});
Y = [p_error_D_10_train, p_error_D_100_train, p_error_D_1000_train];
bar(X, Y, 0.4);
xlabel('Models');
ylabel('P(error)');
title('P(error) vs Model')
%% ============================ Functions ============================= %%
function cost = cost_func(theta, x, label,N)
    h = 1 ./ (1 + exp(-(theta' * x)));	% Sigmoid function
    cost = (-1/N)*( sum(log(h) .* label) + sum(log(1-h) .* (1-label)) );
end