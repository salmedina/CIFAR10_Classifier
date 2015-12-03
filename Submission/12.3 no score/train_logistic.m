function [theta,J]=train_logistic(X, Y, alpha, max_iter)
%This function trains a logistic regression classifier
%through gradient descent
% RETURNS: 1) the parameters theta
%          2) cost J history through iterations
    % m elements; n features
    [m,n] = size(X);
    % add intercept
    theta = zeros(n,1);
    J = zeros(max_iter, 1);

    for i=1:max_iter
        h=sigmoid(X*theta);
        grad = (1/m).*(X'*(h-Y));
        theta = theta - alpha * grad;
        % Calculate J (for testing convergence)
        J(i) =(1/m)*sum(-Y.*log(h) - (1-Y).*log(1-h));
    end
end