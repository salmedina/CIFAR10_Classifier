function [theta,J]=train_multinomial_logistic(X, Y, k, alpha, max_iter)
%This function trains a logistic regression classifier
%through gradient descent
% RETURNS: 1) the parameters theta
%          2) cost J history through iterations
    % m elements; n features
    [m,n] = size(X);
    % add intercept
    X = [ones(m,1), X];
    theta = zeros(k,n+1);
    J = zeros(max_iter, k);

    for t=1:k
        y_t = (Y==t);
        [theta_t,J_t] = train_logistic(X,y_t,alpha,max_iter);
        theta(t,:)=theta_t;
        J(:,t)=J_t;
    end
end