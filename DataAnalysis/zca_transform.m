function X_zca=zca_transform(X, zca)
    X = double(X) - repmat(zca.mean,size(X,1), 1); %shift all features to mean0
    X_zca = zca.eigenvect * diag(1./sqrt(diag(zca.eigenval) + epsilon)) * zca.eigenvect' * X;
end