function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);
             
    % Vars to return
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    % Forward propagation - find J
    A1 = [ones(m,1) X];
    Z2 = A1 * Theta1';
    A2 = [ones(m,1) sigmoid(Z2)];
    Z3 = A2 * Theta2';
    pred_y = sigmoid(Z3);

    [y_POSSIBLE y] = meshgrid(0:num_labels-1, y);
    y = y == y_POSSIBLE;

    J_vals = -(y .* log(pred_y)) - ((1-y) .* log(1 - pred_y));
    J = mean(sum(J_vals, 2));

    J_reg = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
    J = J_reg + J;  


    % Back propagation - find gradients
    for t=1:m
        Delta3 = pred_y(t,:) - y(t,:);
        Delta2 = Delta3 * Theta2 .* [zeros(1,1) sigmoidGradient(Z2(t,:))];
        Delta2 = Delta2(2:end);

        Theta1_grad = Theta1_grad + Delta2' * A1(t,:);
        Theta2_grad = Theta2_grad + Delta3' * A2(t,:);

        Theta1_grad_reg = lambda/m * Theta1;
        Theta2_grad_reg = lambda/m * Theta2;
        Theta1_grad_reg(:,1) = 0;
        Theta2_grad_reg(:,1) = 0;
        Theta1_grad = Theta1_grad_reg + Theta1_grad;
        Theta2_grad = Theta2_grad_reg + Theta2_grad;
    end

    Theta1_grad = 1/m * Theta1_grad;
    Theta2_grad = 1/m * Theta2_grad;


    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];
end



function g = sigmoidGradient(z)
    g = zeros(size(z));
    g = sigmoid(z) .* (1 - sigmoid(z));
end
