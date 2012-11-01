function preprocess()
  training_data = csvread('train.csv', [1,0,30000,785]); %(:,1) is the actual digit value  

  fprintf('preprocessing training data');
  for i=1:30000
    m_orig = reshape(training_data(i, 2:end), [28 28]);
    m = m_orig';
    img = mat2gray(m);

    props = my_regionprops(img);
    theta = props.Orientation

    if (theta < 40 && theta > -40)
      img_new = imrotate(img, -1*theta, 'Fourier', 'crop');
      img_new_matrix = gray2mat(img_new);
      training_data(i, 2:end) = reshape(img_new_matrix', [1 784]);
    end
  end

  save('preprocessed_training.mat', training_data);



  test_data = csvread('test.csv', [1,0, 28000,784]); %(:,1) is the actual digit value  
  fprintf('preprocessing test data');
  for i=1:30000
    m_orig = reshape(test_data(i, :), [28 28]);
    m = m_orig';
    img = mat2gray(m);

    props = my_regionprops(img);
    theta = props.Orientation

    if (theta < 40 && theta > -40)
      img_new = imrotate(img, -1*theta, 'Fourier', 'crop');
      img_new_matrix = gray2mat(img_new);
      test_data(i, :) = reshape(img_new_matrix', [1 784]);
    end
  end

  save('preprocessed_test.mat', test_data);
end

%--- begin stackoverflow answer 
%--- from:http://stackoverflow.com/questions/1711784/computing-object-statistics-from-the-second-central-moments/1712019#1712019

function props = my_regionprops(im)
    cm00 = central_moments(im, 0, 0);
    up20 = central_moments(im, 2, 0) / cm00;
    up02 = central_moments(im, 0, 2) / cm00;
    up11 = central_moments(im, 1, 1) / cm00;

    covMat = [up20 up11 ; up11 up02];
    [V D] = eig( covMat );
    [D order] = sort(diag(D), 'descend');        %# sort cols high to low
    V = V(:,order);

    %# D(1) = (up20+up02)/2 + sqrt(4*up11^2 + (up20-up02)^2)/2;
    %# D(2) = (up20+up02)/2 - sqrt(4*up11^2 + (up20-up02)^2)/2;

    props = struct();
    props.MajorAxisLength = 4*sqrt(D(1));
    props.MinorAxisLength = 4*sqrt(D(2));
    props.Eccentricity = sqrt(1 - D(2)/D(1));
    %# props.Orientation = -atan(V(2,1)/V(1,1)) * (180/pi);      %# sign?
    props.Orientation = -atan(2*up11/(up20-up02))/2 * (180/pi);
end

function cmom = central_moments(im,i,j)
    rawm00 = raw_moments(im,0,0);
    centroids = [raw_moments(im,1,0)/rawm00 , raw_moments(im,0,1)/rawm00];
    cmom = sum(sum( (([1:size(im,1)]-centroids(2))'.^j * ...
                     ([1:size(im,2)]-centroids(1)).^i) .* im ));
end

function outmom = raw_moments(im,i,j)
    outmom = sum(sum( ((1:size(im,1))'.^j * (1:size(im,2)).^i) .* im ));
end
%---- end stack overflow

function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
if ~exist('example_width', 'var') || isempty(example_width) 
  example_width = round(sqrt(size(X, 2)));
end

% Gray Image
colormap(gray);

% Compute rows, cols
[m n] = size(X);
example_height = (n / example_width);

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));

% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
  for i = 1:display_cols
    if curr_ex > m, 
      break; 
    end
    % Copy the patch
    
    % Get the max value of the patch
    max_val = max(abs(X(curr_ex, :)));
    display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                  pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
            reshape(X(curr_ex, :), example_height, example_width) / max_val;
    curr_ex = curr_ex + 1;
  end
  if curr_ex > m, 
    break; 
  end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end

%give each point an (x, y) coordinate, based on indices
% x = index % 28
% y = index / 28 (disregard remainder)

