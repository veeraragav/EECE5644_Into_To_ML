clc
clear

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1; 0];
mu(:,2) = [0.1; 0];
Sigma(:,:,1) = [ 1 -0.9; -0.9, 1 ]; 
Sigma(:,:,2) = [ 1 0.9; 0.9, 1 ]; 
prior = [0.8, 0.2]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= prior(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
dataset = zeros(n,N);


 

% Draw samples from each class pdf
for l = 0:1
    dataset(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
c0_data = [];
c1_data = [];
for i = 1 : N
    if label(i) == 0
        c0_data = [c0_data; dataset(:, i)'];
    end
    if label(i) == 1
        c1_data = [c1_data; dataset(:, i)'];
    end
end
figure(1), clf,
plot(c0_data(:,1),c0_data(:,2),'o'), hold on,
plot(c1_data(:,1),c1_data(:,2),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

%identity covariance matrix
Sigma(:,:,1) = [ 1 0; 0, 1 ]; 
Sigma(:,:,2) = [ 1 0; 0, 1 ];
%liklihood fn
liklihood = @(x,class) mvnpdf(x, mu(class, :), Sigma(:,:,class));

% cost = [0,1,1;1,0,1;1,1,0];

%Decision rule
% gamma = (cost(2,1)-cost(1,1))/(cost(1,2)-cost(2,2)) * prior(1)/prior(2);
discriminantScore = @(x) liklihood(x,2)/ liklihood(x,1);
TP = [];
FP = [];

error_list =[];
g_list = [];
% g = gamma;
min_error = [];
tpr_at_min_error = [];
fpr_at_min_error = [];
threshold_at_min_error = [];
for g = 0:0.025:3
    %classifying class 0 dataset
    c0_classified_as_c0 = [];
    c0_classified_as_c1 = [];

    for i = 1 : 1 : Nc(1)
        label = discriminantScore(c0_data(i, :)) > g;

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
    for i = 1 : 1 : Nc(2)
        label = discriminantScore(c1_data(i, :)) >= g;

        if label == 0
            c1_classified_as_c0 = [c1_classified_as_c0; (c1_data(i, :)) ];
        end

        if label == 1
            c1_classified_as_c1 = [c1_classified_as_c1; (c1_data(i, :)) ];
        end

    end

    %p_c0_given_c1 = size(c1_classified_as_c0,1) / size(c1_data, 1);
    p_c1_given_c0 = size(c0_classified_as_c1,1) / size(c0_data, 1); %fp
    p_c1_given_c1 = size(c1_classified_as_c1,1) / size(c1_data, 1);
    
    error = (size(c1_classified_as_c0,1) + size(c0_classified_as_c1,1)) / N;
    error_list = [error_list error];
    g_list = [g_list g];
    
    if(isempty(min_error))
        min_error = error;
        threshold_at_min_error = g;
        tpr_at_min_error = p_c1_given_c1;
        fpr_at_min_error = p_c1_given_c0;
        
    elseif(error < min_error)
        min_error = error;
        tpr_at_min_error = p_c1_given_c1;
        fpr_at_min_error = p_c1_given_c0;
        threshold_at_min_error = g;
    end
    
    TP = [TP; p_c1_given_c1];
    FP = [FP; p_c1_given_c0];
    
%     figure(3);
%     plot(c0_classified_as_c0(:,1),c0_classified_as_c0(:,2), 'og');
%     hold on;
%     if ~ isempty(c1_classified_as_c1)
%     plot(c1_classified_as_c1(:,1), c1_classified_as_c1(:,2), '+g');
%     hold on;
%     end
%     if ~ isempty(c0_classified_as_c1)
%     plot(c0_classified_as_c1(:,1),c0_classified_as_c1(:,2), 'or');
%     hold on;
%     end
%     plot(c1_classified_as_c0(:,1),c1_classified_as_c0(:,2), '+r');
%     hold off;
%     
%     val = [];
%                                                                                                                                                                                       
%     for h = -10:0.1:20
%         for k = -15:0.1:25   
%         if ceil(log(discriminantScore([h, k])) - log (gamma)) == 0
%             val = [val, [h; k]];
%         end     
%         end
%     end
%     hold on;
%     scatter(val(1,:),val(2, :), 'k.')
    


end
figure(2);
plot( FP', TP');
axis equal;
ylabel('True Positive Probability'), xlabel('False Positive Probability');
title('ROC');

hold on;
plot( fpr_at_min_error, tpr_at_min_error, 'r*');

figure(3);
plot(g_list, error_list);
axis equal;
title('error vs gamma');
xlabel('threshold'); ylabel('error');
