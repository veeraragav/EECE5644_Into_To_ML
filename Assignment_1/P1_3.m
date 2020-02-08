clc;
clear;

n = 2; % number of feature dimensions
N = 10000; % number of iid samples
mu(:,1) = [-0.1; 0];
mu(:,2) = [0.1; 0];
Sigma(:,:,1) = [ 1 -0.9; -0.9, 1 ]; 
Sigma(:,:,2) = [ 1 0.9; 0.9, 1 ]; 
prior = [0.8,0.2]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= prior(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
dataset = zeros(n,N); % save up space

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
mu1hat = mean(c0_data, 1)';
mu2hat = mean(c1_data, 1)';
S1hat = cov(c0_data);
S2hat = cov(c1_data);

% Calculate the between/within-class scatter matrices
Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
Sw = S1hat + S2hat;

% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

% Linearly project the data from both categories on to w
c0_data = c0_data';
c1_data = c1_data';
y0 = w'*c0_data;
y1 = w'*c1_data;

% Plot the data before and after linear projection
figure(1),
subplot(2,1,1), plot(c0_data(1,:),c0_data(2,:),'r*'); hold on;
plot(c1_data(1,:),c1_data(2,:),'bo'); axis equal, 
subplot(2,1,2), plot(y0(1,:),zeros(1,Nc(1)),'r*'); hold on;
plot(y1(1,:),zeros(1,Nc(2)),'bo'); axis equal,

%classification
projected_mean_c0 = w'*mean(c0_data, 2);
projected_mean_c1 = w'*mean(c1_data, 2);


TP = [];
FP = [];

error_list =[];
g_list = [];

min_error = [];
tpr_at_min_error = [];
fpr_at_min_error = [];
threshold_at_min_error = [];

if projected_mean_c1 > projected_mean_c0
   
    for threshold = (min([y0, y1])-2): 0.5 : (max([y0, y1])+2)
%         threshold = -4.407582396994177;
        c0_classified_as_c0 = [];
        c0_classified_as_c1 = [];
        c1_classified_as_c0 = [];
        c1_classified_as_c1 = [];
        
        for ii = 1 : size(y0,2)
            if y0(ii) > threshold
                c0_classified_as_c1 = [c0_classified_as_c1, y0(ii)];
            else
                c0_classified_as_c0 = [c0_classified_as_c0, y0(ii)];
            end    
        end
        
        for ii = 1 : size(y1,2)
            if y1(ii) > threshold
                c1_classified_as_c1 = [c1_classified_as_c1, y1(ii)];
            else
                c1_classified_as_c0 = [c1_classified_as_c0, y1(ii)];
            end    
        end
        
    p_c1_given_c0 = size(c0_classified_as_c1,2) / size(y0, 2); %fp
    p_c1_given_c1 = size(c1_classified_as_c1,2) / size(y1, 2);
    
    error = (size(c1_classified_as_c0,2) + size(c0_classified_as_c1,2)) / N;
    
    error_list = [error_list error];
    g_list = [g_list threshold];
    
    if(isempty(min_error))
        min_error = error;
        threshold_at_min_error = threshold;
        tpr_at_min_error = p_c1_given_c1;
        fpr_at_min_error = p_c1_given_c0;
    elseif(error < min_error)
        min_error = error;
        tpr_at_min_error = p_c1_given_c1;
        fpr_at_min_error = p_c1_given_c0;
        threshold_at_min_error = threshold;
    end
    
    TP = [TP; p_c1_given_c1];
    FP = [FP; p_c1_given_c0];
    
    end
else
    for threshold = min([y0, y1]): 0.1 : max([y0, y1])
        c0_classified_as_c0 = [];
        c0_classified_as_c1 = [];
        c1_classified_as_c0 = [];
        c1_classified_as_c1 = [];
            
        for ii = 1 : size(y0,2)
            if y0(ii) < threshold
                c0_classified_as_c1 = [c0_classified_as_c1, y0(ii)];
            else
                c0_classified_as_c0 = [c0_classified_as_c0, y0(ii)];
            end    
        end
        
        for ii = 1 : size(y1,2)
            if y1(ii) < threshold
                c1_classified_as_c1 = [c1_classified_as_c1, y1(ii)];
            else
                c1_classified_as_c0 = [c1_classified_as_c0, y1(ii)];
            end    
        end
        
        p_c1_given_c0 = size(c0_classified_as_c1,2) / size(y0, 2); %fp
    p_c1_given_c1 = size(c1_classified_as_c1,2) / size(y1, 2);
    
    error = (size(c1_classified_as_c0,1) + size(c0_classified_as_c1,1)) / N;
    
    if(isempty(min_error))
        min_error = error;
        threshold_at_min_error = threshold;
        tpr_at_min_error = p_c1_given_c1;
        fpr_at_min_error = p_c1_given_c0;
    elseif(error < min_error)
        min_error = error;
        tpr_at_min_error = p_c1_given_c1;
        fpr_at_min_error = p_c1_given_c0;
        threshold_at_min_error = threshold;
    end
    
    TP = [TP; p_c1_given_c1];
    FP = [FP; p_c1_given_c0];
    
    
    end
end




figure(2);
plot( FP', TP');
axis equal;
ylabel('True Poisitve Probability'), xlabel('False Poisitve Probability');
title('ROC');
hold on;
plot( fpr_at_min_error, tpr_at_min_error, 'r*');


figure(3);
plot(g_list, error_list);
axis equal;
title('error vs threshold');
xlabel('threshold'); ylabel('error');





