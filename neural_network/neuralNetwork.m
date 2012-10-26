clear;
close all;
clc;

% Sample sizes
num_training_samples = 2000; % Number should be less than 30000
num_cv_samples = 500;
num_test_samples = 500;

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
% Hidden_layer_sizes = [100 150 200];
% Num_hidden_layers = [1 2];
Hidden_layer_sizes = 100;
Num_hidden_layers = 1;



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
if length(Hidden_layer_sizes) > 1 || length(Num_hidden_layers) > 1
    fprintf('Optimizing Parameters...\n');
    [hidden_layer_size num_hidden_layers] = optimizeHiddenLayers (Hidden_layer_sizes, Num_hidden_layers, ...
                                                                  Train_digits, Train_pixels, ...
                                                                  Cv_digits, Cv_pixels, ...
                                                                  input_layer_size, num_labels);
    fprintf('\n\nOptimum hidden layer size: %d\n', hidden_layer_size);
    fprintf('Optimum number of hidden layers: %d\n\n\n', num_hidden_layers);
else
    hidden_layer_size = Hidden_layer_sizes(1);
    num_hidden_layers = Num_hidden_layers(1);
end



%% Training the network
% Train neural network only if data hasn't been previously trained
% If any of the parameters have changed, nn will be re-trained

new_params = struct('num_training_samples', num_training_samples, ...
                'input_layer_size', input_layer_size, ...
                'hidden_layer_size', hidden_layer_size, ...
                'num_hidden_layers', num_hidden_layers, ...
                'num_labels', num_labels);

try
    load ('weightsAndParams.mat');
    fnames = fieldnames(new_params);
    for i=1:nfields(new_params)
        field = fnames{i};

        if new_params.(field) ~= params.(field)
            error ('Foing to catch loop to train network and save weights/params.')
        end
    end

catch
    fprintf ('Training Neural Network... \n');
    THETAS = trainNN(Train_digits, Train_pixels, ...
                               input_layer_size, hidden_layer_size, ...
                               num_hidden_layers, num_labels);
    params = new_params;
    save 'weightsAndParams.mat' params THETAS;
    fprintf ('\n\n');
end




%% Testing the network
fprintf ('Testing Neural Network... \n');
testNN(Test_digits, Test_pixels, THETAS);

