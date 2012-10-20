function logistic_classifier()
    data = csvread('../train.csv', [1,0,30000,785]);
    y_train = data(1:22500,1);
    y_cross = data(22501:end, 1);
    X_train = data(1:22500, 2:end);
    X_cross = data(22501:end, 2:end);
    %%%%test
    % for i=1:size(y)
    %   if (y(i) == 0)
    %     y(i) = 10;
    %   end
    % end
    %%%%end

    fprintf('Optimizing paramaters\n');
    [iter, lambda] = optimizeParameters(X_train, y_train, X_cross, y_cross);

    fprintf('Training Logistic Regression\n');
    [all_theta] = trainLogisticRegression(60);
    
    fprintf('Predicting results for test.csv\n');
    X_test = csvread('../test.csv', [1,0,28000,784]);
    
    m = size(X_test, 1);
    
    % Add ones to the X data matrix
    X_test = [ones(m, 1) X_test];

    A = sigmoid(all_theta * X');
    [p, ip] = max(A);

    csvwrite('logistic_multiclass_results_60iters_05lambda.csv', ip')
end

function [iter, lambda] = optimizeParameters(X_train, y_train, X_cross, y_cross)
  max_accuracy = 0;
  iter = 0;
  lambda = 0;

  X_cross = [ones(size(X_cross, 1), 1) X_cross];
  for l = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    % for i=10:10:100
      [thetas] = trainLogisticRegression(X_train, y_train, 40, l);
      
      A = thetas * X_cross';
      [foo, pred] = max(A);
      
      accuracy = mean(double(pred' == y_cross));
      
      [40, l, accuracy]
      if (accuracy > max_accuracy)
        iter = i;
        lambda = l;
      end
    % end
  end
end

function [all_theta] = trainLogisticRegression(X, y, iter, lambda)
  % function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  %ONEVSALL trains multiple logistic regression classifiers and returns all
  %the classifiers in a matrix all_theta, where the i-th row of all_theta 
  %corresponds to the classifier for label i

  %read the data

  m = size(X, 1);
  n = size(X, 2);
  % lambda = 0.05;
  
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