function [Theta1, Theta2] = trainNN(Digits, Pixels, ...
                                    input_layer_size, hidden_layer_size, ...
                                    num_labels)

    % Useful variables
    max_iter = 50; % Number of minimizing iterations when training
    lambda = 1; % Regularization parameter

    % Make random initial parameters
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

    % Find parameters (minimize cost function)
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, Pixels, Digits, lambda);
    options = optimset('MaxIter', max_iter);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % Reshape variables
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
end


function W = randInitializeWeights(L_in, L_out)
    epsilon_init = 0.12;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end