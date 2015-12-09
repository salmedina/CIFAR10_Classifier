function [Y] = classify2(Model2, X)
% X is the test data, Model is the model learnt during training, 
% and Y contains the predicted labels of the data points in X. 

% The dimensions of X are Ntest × D and dimensions of Y are Ntest × 1. 
% where Ntest is the number of data points in the test data 
% and D is the dimension of each point (number of features).

    %load('Model.mat');
    p = Model2{1};
    Mean = Model2{2};
    Var = Model2{3};

    features = data2features2(X);
    F = size(Mean, 1); % number of features
    C = size(Mean, 2); % number of classes
    N = size(features, 1); % numebr of target samples
    Y = zeros(N, 1);
    %Y = cell(N, 1);
    
    %iterate over target samples
    for k = 1:N 
        Prob = zeros(1, C); % record the probability of each class for this sample
        %iterate over each class
        for j = 1:C
            num = 1;
            %iterate over each feature
            for i = 1:F
                num = num + log(GuassianValue(Mean(i, j), Var(i, j), features(k, i))); % get P(X|Y)
            end
            num = num + log(p(j, 1)); % get the likelihood P(X|Y)P(Y)
            Prob(1, j) = num; % record this probability
        end
        [M, Y(k,1)] = max(Prob(:));
        Y(k,1) = Y(k,1)-1;
        %Y(k, 1) = find(Prob == max(Prob)); 
        %Y{k} = Prob';
    end
end

function [gv] = GuassianValue(mean, variance, xValue)
    exponent = -(((xValue - mean).^2) / (2*variance));
    base = sqrt(2*pi*variance);
    gv = exp(exponent) / base;
end