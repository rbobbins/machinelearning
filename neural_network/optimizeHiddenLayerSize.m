function [opt_hidden_layer_size] = optimizeHiddenLayerSize(Hidden_layer_sizes, ...
                                                    Train_digits, Train_pixels, ...
                                                    Cv_digits, Cv_pixels, ...
                                                    input_layer_size, num_labels)

    Accuracy = zeros(size(Hidden_layer_sizes));

    for i=1:length(Hidden_layer_sizes)
        [Theta1, Theta2] = trainNN(Train_digits, Train_pixels, ...
                                   input_layer_size, Hidden_layer_sizes(i), num_labels);
        Prediction = predict(Theta1, Theta2, Cv_pixels);
        num_correct = length(find(Prediction == Cv_digits));
        Accuracy(i) = num_correct / length(Cv_digits);
    end

    [tmp, i] = max(Accuracy);
    opt_hidden_layer_size = Hidden_layer_sizes(i);

    plot(Hidden_layer_sizes, Accuracy, 'o-');
    xlabel('Hidden layer sizes');
    ylabel('Accuracy');
    ylim([0 1]);
end