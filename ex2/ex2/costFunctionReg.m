function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_copy = theta(2:size(X, 2),1);
z = X*theta;

first = -y' * log(sigmoid(z));
second = (1 - y)' * log(1 - sigmoid(z));
third = lambda/(2*m) *sum(theta_copy.^2);
J = 1/m * sum(first - second) + third;

fourth = sigmoid(z) - y;
%grad(1,1) = 1/m * sum(fourth .*X(:,1));
%for j = 2:size(theta,1)
%    grad(j,1) = 1/m * sum(fourth .* X(:,j)) + lambda/m * theta(j,1);
%end

grad = 1/m * (fourth)'* X + lambda/m * theta';
gradd = (fourth)'* X/m;
grad(1) = gradd(1);

% =============================================================

end
