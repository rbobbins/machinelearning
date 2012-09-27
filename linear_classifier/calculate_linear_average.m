
function calculate_linear_average()
    %================================
    % Calculates the average image for digits 0-9.
    % Saves the result in a matrix - result is persisted in file average.mat
    % Column 1 is the actual digit value
    % Column 2 is the number of samples for this digit
    % Columns 3:786 are the average pixel values for pixels 1-784, read left to right, top to bottom
    %================================

    data = csvread('../train.csv', [1,0,30000,785]); %(:,1) is the actual digit value  

    digits = data(:,1);
    pixels = data(:,2:end);


    index = linspace(0, 9, 10);
    results = zeros(10, 785);
    result = [index', results];

    for k=1:size(digits)
      actual = digits(k);
      row_index = actual + 1;

      result(row_index, 3:end) = result(row_index, 3:end) .+ pixels(k, :);
      result(row_index, 2) = result(row_index, 2) + 1;
    end

    for i=1:10
      result(i, 3:end) = result(i, 3:end) / result(i, 2);
    end

    save average.mat result
end