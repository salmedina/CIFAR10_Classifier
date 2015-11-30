%Example of logistic regression
%Generate Normal data
mu1 = [0 0];
mu2 = [1 1];
sigma = [0.5 0; 0 0.5];
X1 = mvnrnd(mu1, sigma, 6);
X2 = mvnrnd(mu2, sigma, 5);
X = [X1; X2];
Y = [zeros(6,1);ones(5,1)];

shuffle_idx = randperm(m);
X = X(shuffle_idx, :);
Y = Y(shuffle_idx, :);
X = [ones(m,1), X];
MAX_ITER = 3000;
[theta, J]=train_logistic(X,Y,0.1,MAX_ITER);

% Plot J
figure
plot(0:MAX_ITER-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 2)
xlabel('Iteration'); ylabel('Cost')
hold;

%Create a new figure
figure 
% Plot the points with different symbols
pos = find(Y==1);
neg = find(Y==0);
plot(X(pos, 2), X(pos,3), '+')
hold on
plot(X(neg, 2), X(neg, 3), 'o')
hold on
% Plot dividing line
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,3))-2,  max(X(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% Display Results
display([Y, round(sigmoid(X*theta))]);
    