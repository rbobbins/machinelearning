
data = csvread('train.csv', [1,0,30000,784]); %(:,1) is the actual digit value  

digits = data(:,1);
% pixels = data(:,2:end);

% index = linspace(0, 9, 10);
% results = zeros(10, 785);
% result = [index', results]; %first column is digit, second is count, 3:786 are pixel averages

% for k=1:size(digits)
%   actual = digits(k);
%   row_index = actual + 1;

%   result(row_index, 3:end) = result(row_index, 3:end) .+ pixels(k, :);
%   result(row_index, 2) = result(row_index, 2) + 1;
% end

% for i=1:10
%   result(i, 3:end) = result(i, 3:end) / result(i, 2);
% end

% save average.mat result

load('average.mat');
test_data = csvread('test.csv');

total_tries = 0
total_correct = 0

for i=1:size(test_data)
  min_cost = [1000000000000000, 100];
  for j=1:10
    average = result(j, 3:end);
    actual = test_data(i, :);
    cost = sum((average - actual) .^2)/(2*784);
    if (cost < min_cost(1))
      min_cost = [cost, j-1];
    end
  end
  
  total_tries += 1;
  if (min_cost(2) == digits(i))
    total_correct += 1;
  end
end

total_tries
total_correct
total_correct/ total_tries