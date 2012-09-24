clc;
clear all;
close all;

[digits, pixels, m] = load_samples('train.csv', 2000);

zero_digits = [];
for k=1:m
  actual = digits(k);
  if actual == 0
    zero_digits(end+1, 1:784) = pixels(k, :);
  end
end
zero_average = mean(zero_digits, 1);

plot_number(zero_average);