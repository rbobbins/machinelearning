%%  load_data: loads a sequence of 28X28 handwritten number images
%
%   filename: name of file to load data from
%   returns:
%       m: the number of training sampels
%       num_samples: number of samples to load
%       pixels: a mx784 matrix, with each row vector representing one number image
%           Each vector contains a 0-1 pixel value, pixels are stored left to right, 
%           top to bottom
%       digits: a mx1 column vector, with the ith entry corresponding to the 
%       number represented by the ith row vector in pixels

function [digits, pixels, m] = load_samples(filename, num_samples)
    data = csvread(filename, [1, 0, num_samples, 784]); %(:,1) is the actual digit value  

    digits = data(:,1);
    pixels = data(:,2:end);
    m = length(digits);
end