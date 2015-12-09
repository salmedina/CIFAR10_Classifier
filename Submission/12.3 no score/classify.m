function [Y] = classify(Model, X)
% X is the test data, Model is the model learnt during training, 
% and Y contains the predicted labels of the data points in X. 

% The dimensions of X are Ntest ? D and dimensions of Y are Ntest ? 1. 
% where Ntest is the number of data points in the test data 
% and D is the dimension of each point (number of features).

    
end

function [gv] = GuassianValue(mean, variance, xValue)
    exponent = -(((xValue - mean).^2) / (2*variance));
    base = sqrt(2*pi*variance);
    gv = exp(exponent) / base;
end