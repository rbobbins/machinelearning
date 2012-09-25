

load('average.mat');
test_data = csvread('test.csv', [1, 0, 28000, 783]);

total_tries = 0
total_correct = 0

digit_guesses = zeros(28000, 1);

for i=1:size(test_data)
  min_cost = [1000000000000000, 100]; %[cost, prediction]
  actual = test_data(i, :);

  for j=1:10
    average = result(j, 3:end);
    
    cost = sum((average - actual) .^2)/(2*784);
    if (cost < min_cost(1))
      min_cost = [cost, j-1];
    end
  end

  digit_guesses(i) = min_cost(2);
end

csvwrite('prediction.csv', digit_guesses);