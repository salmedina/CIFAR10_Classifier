%Generate random data around (0,0) and (2,2)
mu1 = [0 0];
mu2 = [2 2];
mu3 = [-1 3];
sigma = [1 0; 0 1];
X1 = mvnrnd(mu1, sigma, 60000);
X2 = mvnrnd(mu2, sigma, 50000);
X3 = mvnrnd(mu3, sigma, 70000);
X = [X1; X2; X3];
[n,d] = size(X);
X = X(randperm(n), :);

%Call K-means
[idx, centroids] = k_means(X, 3);
size(idx)
size(centroids)
%Plot
clrs = idx;
clrs(idx==1) = 'y';
clrs(idx==2) = 'm';
clrs(idx==3) = 'b';
scatter(X(:,1), X(:,2), 25, clrs, 'filled');