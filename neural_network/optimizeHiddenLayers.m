function [opt_hidden_layer_size opt_num_hidden_layers] = optimizeHiddenLayers(Hidden_layer_sizes, Num_hidden_layers, ...
                                                                                Train_digits, Train_pixels, ...
                                                                                Cv_digits, Cv_pixels, ...
                                                                                input_layer_size, num_labels)

    [HL_SIZES NUM_HL] = meshgrid (Hidden_layer_sizes, Num_hidden_layers);
    ACCURACY = zeros(size(HL_SIZES));

    for i=1:length(Num_hidden_layers)
        for j=1:length(Hidden_layer_sizes)
            fprintf ('Hidden layer size: %d\n', Hidden_layer_sizes(j));
            fprintf ('Num hidden layers: %d\n', Num_hidden_layers(i));
            THETAS = trainNN(Train_digits, Train_pixels, ...
                                       input_layer_size, Hidden_layer_sizes(j), ...
                                       Num_hidden_layers(i), num_labels);
            Prediction = predict(THETAS, Cv_pixels);
            num_correct = length(find(Prediction == Cv_digits));
            ACCURACY(i,j) = num_correct / length(Cv_digits);
        end
    end

    [tmp, js] = max(ACCURACY);
    [tmp, i] = max(tmp);
    j = js(i);
    opt_hidden_layer_size = Hidden_layer_sizes(i);
    opt_num_hidden_layers = Num_hidden_layers(j);

    fprintf ('\nAccuracy:\n(Rows: varying size of hidden layer.) Values:\n');
    disp (Hidden_layer_sizes);
    fprintf ('(Cols: varying the number of hidden layers.) Values:\n');
    disp (Num_hidden_layers);
    fprintf ('\n');
    disp (ACCURACY);

    Legend_entries = cell (size(Num_hidden_layers));
    for i=1:length(Num_hidden_layers)
        Legend_entries{i} = sprintf ('%d hidden layers', Num_hidden_layers(i));
    end
    plot (Hidden_layer_sizes, ACCURACY, 'o-');
    legend (Legend_entries, 'Location', 'NorthWest')
    xlabel ('Hidden layer sizes');
    ylabel ('Number of hidden layers');
    
    ylim([0 1]);
end