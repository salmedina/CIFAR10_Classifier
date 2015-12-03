function [ Y ] = tmp( features, mean, var, p )
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
    F = size(mean, 1); % number of features
    C = size(mean, 2); % number of classes
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
                num = num + log(GuassianValue(mean(i, j), var(i, j), features(k, i))); % get P(X|Y)
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