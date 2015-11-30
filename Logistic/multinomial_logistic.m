%Example of logistic regression
%Generate Normal data
mu1 = [0 0];
mu2 = [2 2];
mu2 = [1- 3];
sigma = [0.5 0; 0 0.5];
X1 = mvnrnd(mu1, sigma, 6);
X2 = mvnrnd(mu2, sigma, 5);
X3 = mvnrnd(mu7, sigma, 7);
X = [X1; X2; X3];
Y = [ones(6,1);2.*ones(5,1);3.*ones(7,1)];

shuffle_idx = randperm(m);
X = X(shuffle_idx, :);
Y = Y(shuffle_idx, :);

%Add intercept
X = [ones(m,1), X];





figure %create a new figure
pos = find(Y==1);
neg = find(Y==0);
plot(X(pos, 2), X(pos,3), '+')
hold on
plot(X(neg, 2), X(neg, 3), 'o')
hold on

% Plot dividing line
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
% Calculate the decision boundary lines for each class
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off


% Display J
display(J)
display([Y, round(sigmoid(X*theta))]);
    