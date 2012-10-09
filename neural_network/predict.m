function P = predict(Theta1, Theta2, X)
    m = size(X, 1);
    num_labels = size(Theta2, 1);

    % you need to return the following variables correctly 
    P = zeros(size(X, 1), 1);

    H1 = sigmoid([ones(m, 1) X] * Theta1');
    H2 = sigmoid([ones(m, 1) H1] * Theta2');
    [dummy, P] = max(H2, [], 2);
    P = P - ones(size(P)); % Make everything '0-indexed'
end