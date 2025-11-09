function [coeff, score] = manual_pca(X, k)
% MANUAL_PCA Compute PCA using covariance + eigen decomposition
% X : zero-mean data matrix (samples x features) - function expects zero-mean
% k : number of principal components to return
% Returns coeff (features x k) and score (samples x k)

% Center the data
Xc = X - mean(X,1);

% Covariance
C = cov(Xc);

% Eigen decomposition
[V, D] = eig(C);

% Sort eigenvectors by descending eigenvalues
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);

% Return top-k
coeff = V(:, 1:k);
score = Xc * coeff;
end
