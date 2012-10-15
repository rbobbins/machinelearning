clear;
close all;
clc;

% Sample sizes
num_training_samples = 240; % Number should be less than 30000
num_cv_samples = 80;
num_test_samples = 80;

% Fixed variables
input_layer_size = 28 * 28; % Input: each pixel is a feature
num_labels = 10; % Output: each digit is a label
% hidden_layer_size - will be optimized over
% num_hidden_layers - will be optimized over
%   each hidden layer will have the same number of children/nodes

% Variables to optimize over
% For each of these variables:
    % if a vector is supplied - it will optimize that variable over the supplied values in the vector
    % is a scalar is supplied - it will NOT perform optimization
% Hidden_layer_sizes = [25 50 100 200];
Hidden_layer_sizes = 200;



%% Check gradients
% fprintf('Testing backpropagation algorithm against numerical gradient calculations\n');
% lambda = 1;
% checkNNGradients(lambda);
% break;



%% Data
% Training data
train_data_indicies = [1, 0, num_training_samples, input_layer_size];
Train_data = csvread('../train.csv', train_data_indicies);
Train_digits = Train_data(:,1);
Train_pixels = Train_data(:,2:end);

% Cross validation data
cv_data_indicies = [num_training_samples + 1, 0, num_training_samples + num_cv_samples, input_layer_size];
Cv_data = csvread('../train.csv', cv_data_indicies);
Cv_digits = Cv_data(:,1);
Cv_pixels = Cv_data(:,2:end);

% Test data
num_prior_samples = num_training_samples + num_cv_samples;
test_data_indicies = [num_prior_samples + 1, 0, num_prior_samples + num_test_samples, input_layer_size];
Test_data = csvread('../train.csv', test_data_indicies);
Test_digits = Test_data(:,1);
Test_pixels = Test_data(:,2:end);



%% Optimizing params
fprintf('\n\nOptimizing Parameters...\n');
if length(Hidden_layer_sizes) > 1
    hidden_layer_size = optimizeHiddenLayerSize(Hidden_layer_sizes, ...
                                                Train_digits, Train_pixels, ...
                                                Cv_digits, Cv_pixels, ...
                                                input_layer_size, num_labels);
else
    hidden_layer_size = Hidden_layer_sizes(1);
end
fprintf('Optimum hidden layer size: %d\n', hidden_layer_size);



%% Training the network
% Train neural network only if data hasn't been previously trained
% If any of the parameters have changed, nn will be re-trained

new_params = struct('num_training_samples', num_training_samples, ...
                'input_layer_size', input_layer_size, ...
                'hidden_layer_size', hidden_layer_size, ...
                'num_labels', num_labels);

try
    load('weightsAndParams.mat');
    for i=1:nfields(new_params)
        field = fieldnames(new_params){i};

        if new_params.(field) ~= params.(field)
            fprintf('\n\nTraining Neural Network... \n');
            [Theta1, Theta2] = trainNN(Train_digits, Train_pixels, ...
                                       input_layer_size, hidden_layer_size, num_labels);
            params = new_params;
            save 'weightsAndParams.mat' params Theta1 Theta2;
        end
    end

catch
    fprintf('\n\nTraining Neural Network... \n');
    [Theta1, Theta2] = trainNN(Train_digits, Train_pixels, ...
                               input_layer_size, hidden_layer_size, num_labels);
    params = new_params;
    save 'weightsAndParams.mat' params Theta1 Theta2;
end



%% Testing the network
fprintf('\n\nTesting Neural Network... \n');
testNN(Test_digits, Test_pixels, Theta1, Theta2);

