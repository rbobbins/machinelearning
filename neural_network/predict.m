function P = predict(THETAS, X)             
    m = size(X, 1);
    num_labels = size(THETAS{end}, 1);

    % you need to return the following variables correctly 
    P = zeros(size(X, 1), 1);

    H = X;
    for i=1:length(THETAS)
        Theta = THETAS{i};
        H = sigmoid([ones(m, 1) H] * Theta');
    end
    [dummy, P] = max(H, [], 2);
    P = P - ones(size(P)); % Make everything '0-indexed'
end