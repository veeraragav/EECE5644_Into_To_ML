clc
clear
n = 2; % number of feature dimensions
N = 1000; % number of iid samples
prior = [0.7, 0.3];  % class priors for labels 0 and 1 respectively
mu0(:,1) = [1;6];
mu0(:,2) = [-1;2];

Sigma0(:,:,1) = [1 0.9; 0.9 1]; 
Sigma0(:,:,2) = [2 0.5; 0.5 2];

alpha0 = [0.6,0.4];


mu1(:,2) = [6;8];
mu1(:,1) = [5;4];

Sigma1(:,:,1) = [1 -0.9; -0.9 1]; 
Sigma1(:,:,2) = [2 0; 0 2];

alpha1 = [0.55, 0.45];


%sample generation
uniform = rand(1,N);

c0_data = [];
c1_data = [];
for ii = 1 : N
    if uniform(ii) < prior(1)
        random = rand();
        if random < alpha0(1) %first gaussian in gmm0
            data = mvnrnd(mu0(:,1),Sigma0(:,:,1))';
            c0_data = [c0_data data];
   
        else %3nd gaussian in gmm0
            data = mvnrnd(mu0(:,2),Sigma0(:,:,2))';
            c0_data = [c0_data data];
        end
    
    else
        random = rand();
        if random < alpha1(1) %first gaussian in gmm1
            data = mvnrnd(mu1(:,1),Sigma1(:,:,1))';
            c1_data = [c1_data data];
       
        else %3nd gaussian in gmm1
            data = mvnrnd(mu1(:,2),Sigma1(:,:,2))';
            c1_data = [c1_data data];
        end
        
        
    end
end

%display sample data
figure(1);
plot(c0_data(1,:),c0_data(2, :),'ob'), hold on,
plot(c1_data(1,:),c1_data(2, :),'+m'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'),


%liklihood fn
liklihood0 = @(x) alpha0(1)*mvnpdf(x,mu0(:,1)',Sigma0(:,:,1)) + alpha0(2)*mvnpdf(x,mu0(:,2)',Sigma0(:,:,2)) ;
liklihood1 = @(x) alpha1(1)*mvnpdf(x,mu1(:,1)',Sigma1(:,:,1)) + alpha0(2)*mvnpdf(x,mu1(:,2)',Sigma1(:,:,2)) ;
%cost matrix
cost = [ 0, 1; 1, 0];
gamma = (cost(2,1)-cost(1,1))/(cost(1,2)-cost(2,2)) * prior(1)/prior(2);
g = gamma;
TP = [];
FP = [];
discriminantScore = @(x) liklihood1(x)/ liklihood0(x);
Nc = [size(c0_data, 2), size(c1_data, 2)];


%classifying class 0 dataset
c0_classified_as_c0 = [];
c0_classified_as_c1 = [];

for i = 1 : 1 : Nc(1)
    label = discriminantScore(c0_data( :, i)') >= g;

    if label == 0
        c0_classified_as_c0 = [c0_classified_as_c0, (c0_data( :, i)) ];
    end

    if label == 1
        c0_classified_as_c1 = [c0_classified_as_c1, (c0_data(:, i)) ];
    end

end

%classifying class 1 dataset
c1_classified_as_c0 = [];
c1_classified_as_c1 = [];
for i = 1 : 1 : Nc(2)
    label = discriminantScore(c1_data(:, i)') >= g;

    if label == 0
        c1_classified_as_c0 = [c1_classified_as_c0, (c1_data(:, i)) ];
    end

    if label == 1
        c1_classified_as_c1 = [c1_classified_as_c1, (c1_data(:, i)) ];
    end

end


% p_c1_given_c0 = size(c0_classified_as_c1,2) / size(c0_data, 2); %fp
% p_c1_given_c1 = size(c1_classified_as_c1,2) / size(c1_data, 2);
% TP = [TP; p_c1_given_c1];
% FP = [FP; p_c1_given_c0];


%finding decision boundary
val = [];
for h = -10:0.1:20
    for k = -15:0.1:25

    if ceil(log(discriminantScore([h, k])) - log (gamma)) == 0
        val = [val, [h; k]];
    end

    end
end
    

%plotting the results
figure(2);
scatter(val(1,:),val(2, :), 'k.', 'DisplayName','Decision Boundary');


hold on;
plot(c0_classified_as_c0(1,:),c0_classified_as_c0(2, :), 'og', 'DisplayName', 'Class0 classified as Class0');


hold on;
plot(c1_classified_as_c1(1,:), c1_classified_as_c1(2, :), '+g', 'DisplayName', 'Class1 classified as Class1');


hold on;
if ~isempty(c0_classified_as_c1)
plot(c0_classified_as_c1(1,:),c0_classified_as_c1(2, :), 'or', 'DisplayName', 'Class0 classified as Class1');
hold on;
end


hold on;
if ~isempty(c1_classified_as_c0)
plot(c1_classified_as_c0(1,:),c1_classified_as_c0(2, :), '+r', 'DisplayName', 'Class1 classified as Class0');

hold on;
end

legend;
title('Data after classification'),
xlabel('x_1'), ylabel('x_2'),

axis equal;
hold off;

