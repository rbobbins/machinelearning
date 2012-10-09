function [Digits, Prediction] = testNN(Theta1, Theta2, ...
              num_test_samples, num_prior_samples, input_layer_size)
    Data = csvread('../train.csv', [num_prior_samples, 0, ...
                                    num_prior_samples + num_test_samples, input_layer_size]);
    Digits = Data(:,1);
    Pixels = Data(:,2:end);

    Prediction = predict(Theta1, Theta2, Pixels);
    Incorrect = find(Prediction ~= Digits);
    fprintf('Percent correct: %.2f\n', ...
            (num_test_samples - length(Incorrect))/num_test_samples);

    fprintf('\nPrediction: \tActual:\n');
    for i=Incorrect
        fprintf('%d\t\t%d\n', Prediction(i), Digits(i));
    end
end