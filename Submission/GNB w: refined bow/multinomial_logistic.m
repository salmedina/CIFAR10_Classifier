function [prob, Y]=multinomial_logistic(X, multi_theta)
    simgoid = @(z) 1.0/(1.0+exp(-z));
    %add Intercept
    [m,n]=size(X);
    X = [ones(m,1), X];
    [prob, Y] = max(sigmoid(X*multi_theta'),[],2);
end