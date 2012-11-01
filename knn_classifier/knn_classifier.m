function knn_classifier()
    training_data = csvread('../train.csv', [1,0,30000,785]);

    y_train = training_data(:, 1);   
    X_train = training_data(:, 2:end);

    X_test = csvread('../test.csv', [1,0,28000,784]); %28000

    % k = optimize_k(X_train, y_train);
    k = 15;
    y_test = zeros(size(X_test, 1), 1);

    for i = 1:size(X_test, 1)
      y_test(i) = nearest_neighbors(X_train, y_train, X_test(i, :), k)
    end
    csvwrite('knn_classifier_k20.csv', y_test);
end

function [best_k] = optimize_k(X_train_orig, y_train_orig)
  X_train = X_train_orig(1:27999, :);
  X_cross = X_train_orig(28000:end, :);

  y_train = y_train_orig(1:27999, :);
  y_cross = y_train_orig(28000:end, :);

  k = 15;
  best_k = 0;
  best_accuracy = 0;
  for k = 1:20
    y_test = zeros(size(X_cross, 1), 1);
    
    for i = 1:size(X_cross, 1)
      y_test(i) = nearest_neighbors(X_train, y_train, X_cross(i, :), k);
    end

    accuracy = mean(double(y_test == y_cross));
    [k, accuracy]
    if (accuracy > best_accuracy)
      best_k = k;
      best_accuracy = accuracy;
    end
  end

end

function [y] = nearest_neighbors(X_train, y_train, x_test, K)
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

