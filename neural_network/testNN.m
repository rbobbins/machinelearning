function [Digits, Prediction] = testNN(Test_digits, Test_pixels, ...
                                       Theta1, Theta2)

    Prediction = predict(Theta1, Theta2, Test_pixels);
    Incorrect = find(Prediction ~= Test_digits);
    fprintf('Percent correct: %.2f\n', ...
            (length(Test_digits) - length(Incorrect))/length(Test_digits));

    fprintf('\nPrediction: \tActual:\n');
    for i=Incorrect
        fprintf('%d\t\t%d\n', Prediction(i), Test_digits(i));
    end
end