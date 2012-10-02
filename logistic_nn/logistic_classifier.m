function logistic_classifier()
    [all_theta] = trainLogisticRegression(50);
    X = csvread('../test.csv', [1,0,28000,784]);
    
    m = size(X, 1);
    num_labels = size(all_theta, 1);
    
    % Add ones to the X data matrix
    X = [ones(m, 1) X];

    A = sigmoid(all_theta * X');
    [p, ip] = max(A);

    csvwrite('logistic_multiclass_results.csv', ip')
end

function [all_theta] = trainLogisticRegression(iter)
  % function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  %ONEVSALL trains multiple logistic regression classifiers and returns all
  %the classifiers in a matrix all_theta, where the i-th row of all_theta 
  %corresponds to the classifier for label i

  %read the data
  data = csvread('../train.csv', [1,0,30000,785]);
  y = data(:,1);
  X = data(:,2:end);

  m = size(X, 1);
  n = size(X, 2);
  lambda = 0.1;
  
  % Add ones to the X data matrix
  X = [ones(m, 1) X];

  all_theta = zeros(10, n + 1);
  for c=1:10
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', iter);
    theta = fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
    all_theta(c,:) = theta;
  end
end


function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  %regularization

  m = length(y); % number of training examples

  % 1.3.1 Vectorized, unregularized cost function
  h = sigmoid(X * theta);
  J = (1/m) * sum((-y .* log(h)) - ((1-y) .* log(1-h)));

  % 1.3.2 Vectorized, unregularized gradient
  % grad = 1/m * X' * (h-y);

  % 1.3.3 Regularized cost function 
  J = J + lambda / (2 * m) * sum(theta(2:end).^2);

  % 1.3.3 Regularized gradient
  temp = theta;
  temp(1) = 0;
  grad = 1/m * X' * (h-y) + (temp .* lambda / m);

  % =============================================================
  grad = grad(:);

end

function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

  g = 1.0 ./ (1.0 + exp(-z));
end