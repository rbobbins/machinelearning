clear;
close all;
clc;

input_layer_size = 28 * 28; % Input: each pixel is a feature
hidden_layer_size = 25;
num_labels = 10; % Output: each digit is a label

num_training_samples = 5000; % Number should be less than 30000
num_test_samples = 50;

lambda = 1;


%% Check gradients
% fprintf('Testing backpropagation algorithm against numerical gradient calculations\n');
% checkNNGradients(lambda);


%% Training the network
% Train neural network only if data hasn't been previously trained
% If any of the parameters have changed, nn will be re-trained

new_params = struct('num_training_samples', num_training_samples, ...
                'lambda', lambda, ...
                'input_layer_size', input_layer_size, ...
                'hidden_layer_size', hidden_layer_size, ...
                'num_labels', num_labels);

try
    load('weightsAndParams.mat');
    for i=1:nfields(new_params)
        field = fieldnames(new_params){i};

        if new_params.(field) ~= params.(field)
            fprintf('\n\nTraining Neural Network... \n');
            [Theta1, Theta2] = trainNN(num_training_samples, lambda, ...
                                       input_layer_size, hidden_layer_size, num_labels);
            params = new_params;
            save 'weightsAndParams.mat' params Theta1 Theta2;
        end
    end

catch
    fprintf('\n\nTraining Neural Network... \n');
    [Theta1, Theta2] = trainNN(num_training_samples, lambda, ...
                               input_layer_size, hidden_layer_size, num_labels);
    params = new_params;
    save 'weightsAndParams.mat' params Theta1 Theta2;
end


%% Testing the network
fprintf('\n\nTesting Neural Network... \n');
num_prior_samples = num_training_samples;
testNN(Theta1, Theta2, num_test_samples, num_prior_samples, input_layer_size);

