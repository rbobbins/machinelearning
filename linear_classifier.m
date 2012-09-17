clc;
clear all;
close all;

data = csvread('train.csv', [1,0,100,784]); %(:,1) is the actual digit value  

digits = data(:,1);
pixels = data(:,2:end);

zero_digits = [];
for k=1:100
  actual = digits(k);
  if actual == 1
    zero_digits(end+1, 1:784) = pixels(k, :);
  end
end

zero_average = mean(zero_digits, 1);

figure();
axis square;
xlim([1 28]);
ylim([1 28]);

hold on;
for i=0:27
  for j=1:28
    rgb = zero_average(28*i + j);
    if (rgb != 0)
      hold on;
      color = (1 - rgb/256) * [1,1,1];
      plot(j,28-i, "markeredgecolor", color, "markerfacecolor", color, "marker", 'o');
    end
  end
end 