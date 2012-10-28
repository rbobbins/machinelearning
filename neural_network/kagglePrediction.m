%% Make predictions on competition test set
fprintf ('Making predictions...\n');

load ('weightsAndParams.mat');
test_data_indicies = [1, 0, 28000, params.input_layer_size-1];
Test_data = csvread('../test.csv', test_data_indicies);

predictions = predict(THETAS, Test_data);
disp (predictions(1:20));
fprintf ('...\n');
csvwrite (sprintf('neural_network_%d_hl_with_%d_nodes.csv', params.num_hidden_layers, params.hidden_layer_size), predictions);
break;  