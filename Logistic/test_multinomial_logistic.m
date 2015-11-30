%Example of logistic regression
%Generate Normal data
mu1 = [0 0];
mu2 = [2 2];
mu3 = [-2 2];
sigma = [0.5 0; 0 0.5];

X1 = mvnrnd(mu1, sigma, 6);
X2 = mvnrnd(mu2, sigma, 5);
X3 = mvnrnd(mu3, sigma, 7);

X = [X1; X2; X3];
Y = [ones(6,1);2.*ones(5,1);3.*ones(7,1)];

[m,n]=size(X);
shuffle_idx = randperm(m);
X = X(shuffle_idx, :);
Y = Y(shuffle_idx, :);

%Add intercept
X = [ones(m,1), X];

%TRAIN
k=3; %number of classes
multi_theta = zeros(3,3);
alpha = 0.1;
max_iter = 3000;
for t=1:k
    display(t)
    y_t = (Y==t)
    [theta_t,J_t] = train_logistic(X,y_t,alpha,max_iter);
    multi_theta(t,:)=theta_t
end

%TEST
[prob, Y_out] = max(sigmoid(X*multi_theta'),[],2);

figure %create a new figure
pluses = find(Y==1);
circles = find(Y==2);
squares = find(Y==3);
plot(X(pluses, 2), X(pluses,3), '+')
hold on
plot(X(circles, 2), X(circles, 3), 'o')
hold on
plot(X(squares, 2), X(squares, 3), '*')
hold on

% Plot decision boundaries
% Only need 2 points to define a line, so choose two endpoints
% Calculate the decision boundary lines for each class
plot_x = [min(X(:,2))-1,  max(X(:,2))+1];
plot_x_size = size(plot_x)
plot_y = zeros(k,2)
for t=1:k
    plot_y(t,:) = (-1./multi_theta(t,3)).*(multi_theta(t,2).*plot_x +multi_theta(t,1));
    plot(plot_x, plot_y(t,:));
    hold on
end

legend('Class 1', 'Class 2', 'Class 3', 'Limit 1', 'Limit 2', 'Limit 3')
hold off
display(plot_y)

% Display J
%display(J)
display([Y, Y_out]);
    