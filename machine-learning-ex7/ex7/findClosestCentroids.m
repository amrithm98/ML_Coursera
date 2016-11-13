function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m=size(X,1);
% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

cost_complete=zeros(m,K); % m X K matrix with cost of each example for every k
for k=1:K
	centroid_k=ones(m,1)*centroids(k,:); % m copies of centroid k (with n features)
	cost_complete(:,k)=sum((X-centroid_k).^2,2); %error value computed 
endfor
[min,idx]=min(cost_complete'); % min of k columns for ith example
% =============================================================

end
