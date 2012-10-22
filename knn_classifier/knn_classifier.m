function knn_classifier()
    training_data = csvread('../train.csv', [1,0,300,785]);

    y_train = training_data(:, 1);   
    X_train = training_data(:, 2:end);

    X_test = csvread('../test.csv', [1,0,28,784]); %28000

    k = 15;
    y_test = zeros(size(X_test, 1), 1);

    for i = 1:size(X_test, 1)
      y_test(i) = nearest_neighbors(X_train, y_train, X_test(i, :), k)
    end
    % csvwrite('knn_classifier_k20.csv', y_test);
end

function [y] = nearest_neighbors_improved(X_train, y_train, x_test, K)
  %calculate euclidean distance from x_test to every example in X_train
  m = size(X_train, 1);
  d = X_train - repmat(x_test, m, 1);
  d = sqrt(sum(d .^ 2, 2));

  %k closest neighbors to x_test
  prediction = zeros(K, 1);
  for k = 1:K
    [dist, index] = min(d);           %finds the k smallest distance, and its index   
    prediction(k) = y_train(index);

    d(index) = inf; %so it won't count as min during the next iteration
  end

  y = mode(prediction);
end






% function [y] = nearest_neighbor(X_train, y_train, x_test, K)
%   d = euclidian_distances(X_train, x_test);

%   prediction = zeros(K, 1);
%   for k = 1:K
%     [dist, index] = min(d); %finds the k smallest distance, and its index   
%     prediction(k) = y_train(index);

%     d(index) = inf; %so it won't count as min during the next iteration
%   end

%   y = mode(prediction);
% end

% function [d] = euclidian_distances(X_train, x_test)
%   %finds the distance from every x_test to each entry in X_train
%   m = size(X_train, 1);   %number of training examples  
%   d = zeros(m, 1);        %distance from testcase to every training example

%   % d = sqrt(sum((X_train .- x_test) .^2) );
%   for i = 1:m
%     d(i) = sqrt(sum((X_train(i, :) - x_test) .^ 2));
%   end
% end