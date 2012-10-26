function THETAS = trainNN(Digits, Pixels, ...
                                    input_layer_size, hidden_layer_size, ...
                                    num_hidden_layers, num_labels)

    % Useful variables
    max_iter = 40; % Number of minimizing iterations when training
    lambda = 1; % Regularization parameter

    % Make random initial parameters
    initial_nn_params = [];

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_nn_params = [initial_Theta1(:)];
    for i=1:num_hidden_layers-1
        initial_ThetaNext = randInitializeWeights(hidden_layer_size, hidden_layer_size);
        initial_nn_params = [initial_nn_params; initial_ThetaNext(:)];
    end
    initial_ThetaEnd = randInitializeWeights(hidden_layer_size, num_labels);
    initial_nn_params = [initial_nn_params; initial_ThetaEnd(:)];

    % Find parameters (minimize cost function)
    costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       num_labels, ...
                                       num_hidden_layers, Pixels, Digits, lambda);
    options = optimset('MaxIter', max_iter);
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % Reshape variables
    % Reshape nn_params back into the weight matrices for our neural network
    THETAS = cell(1, num_hidden_layers+1);

    THETAS{1} = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    num_used_entries = hidden_layer_size * (input_layer_size + 1);

    for i=1:num_hidden_layers-1
        THETAS{i+1} = reshape(nn_params((num_used_entries+1):(num_used_entries + hidden_layer_size*(hidden_layer_size+1))), ...
                     hidden_layer_size, (hidden_layer_size + 1));
        num_used_entries = num_used_entries + hidden_layer_size*(hidden_layer_size+1);
    end

    THETAS{end} = reshape(nn_params((1 + num_used_entries):end), num_labels, (hidden_layer_size + 1));
end


function W = randInitializeWeights(L_in, L_out)
    epsilon_init = 0.12;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end