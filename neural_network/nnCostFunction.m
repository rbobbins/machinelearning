function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

    num_hidden_layers = 1;

    % Setup some useful variables
    m = size(X, 1);

    % Reshape nn_params back into the weight matrices for our neural network
    THETAS = cell(1, num_hidden_layers+1);
    THETAS{1} = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    for i=1:num_hidden_layers
        THETAS{i+1} = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    end
             
    % Vars to return
    J = 0;
    THETA_GRADS = cell(size(THETAS));
    THETA_GRADS_REG = cell(size(THETAS));
    for i=1:length(THETAS)
        THETA_GRADS{i} = zeros(size(THETAS{i}));
        THETA_GRADS_REG{i} = zeros(size(THETAS{i}));
    end


    % Forward propagation - find J
    AS = cell(1, num_hidden_layers+2);
    ZS = cell(1, length(AS));

    AS{1} = [ones(m,1) X];
    for i=2:length(AS)-1
        ZS{i} = AS{i-1} * THETAS{i-1}';
        AS{i} = [ones(m,1) sigmoid(ZS{i})];
    end
    ZS{end} = AS{end-1} * THETAS{end}';
    AS{end} = sigmoid(ZS{end}); % Predicted Y value

    [y_POSSIBLE y] = meshgrid(0:num_labels-1, y);
    y = y == y_POSSIBLE;

    J_vals = -(y .* log(AS{end})) - ((1-y) .* log(1 - AS{end}));
    J = mean(sum(J_vals, 2));

    J_reg = lambda/(2*m) * (sum(sum(THETAS{1}(:,2:end).^2)) + sum(sum(THETAS{2}(:,2:end).^2)));
    J = J_reg + J;  


    % Back propagation - find gradients
    DELTAS = cell(1, length(AS));

    for t=1:m
        DELTAS{end} = AS{end}(t,:) - y(t,:);
        for i = length(AS) - (1:length(AS)-2)
            DELTAS{i} = DELTAS{i+1} * THETAS{i} .* [zeros(1,1) sigmoidGradient(ZS{i}(t,:))];
            DELTAS{i} = DELTAS{i}(2:end);
        end

        for i=1:length(THETA_GRADS)
            THETA_GRADS{i} = THETA_GRADS{i} + DELTAS{i+1}' * AS{i}(t,:);
            THETA_GRADS_REG{i} = lambda/m * THETAS{i};
            THETA_GRADS_REG{i}(:,1) = 0;
            THETA_GRADS{i} = THETA_GRADS{i} + THETA_GRADS_REG{i};
        end
    end

    for i=1:length(THETA_GRADS)
        THETA_GRADS{i} = 1/m * THETA_GRADS{i};
    end


    % Unroll gradients
    grad = [THETA_GRADS{1}(:) ; THETA_GRADS{2}(:)];
end



function g = sigmoidGradient(z)
    g = zeros(size(z));
    g = sigmoid(z) .* (1 - sigmoid(z));
end
