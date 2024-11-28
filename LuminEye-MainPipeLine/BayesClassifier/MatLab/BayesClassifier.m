clear all;
close all;
clc;




%% Iris region detection --------------------------------------------------

% Read eye image
img = imread('I2head_5_left.png');


% 1 height
%2 width



% Load likelihood values stored into a 64-bin histogram
hist = dlmread('ScleraSkinIris.txt');

% Define a threshold value for binarisation
thresh = 0.7;

% Retain only the red channel image
imgR = img(:, :, 1);

% Perform Gaussian filtering
imgR_gauss = imgaussfilt(imgR, 1);


figure; 
imshow(imgR_gauss);

% Initialise binary image having the same size as the red channel image
imgR_bin = zeros(size(imgR, 1),size(imgR, 2));

% Iterate through the red channel image to threshold it
for i = 1:size(imgR, 1)
    for j = 1:size(imgR, 2)
        
        % Compute the histogram bin number from which to read the
        % likelihood value
        binNo = (imgR_gauss(i, j) + 1) / 4;

        
        % Read the likelihood value from the histogram bin
        likelihood = hist(binNo);
        
        % Binarise the image
        if likelihood >= thresh
            
            imgR_bin(i, j) = 1;
            
        end
        
        
    end
end

figure;
imshow(imgR_bin, [0, 1]);
title('Bayes Classification');


%% Iris centroid estimation -----------------------------------------------

clc
% Remove small blobs from binary image
%imgR_bin = bwareaopen(imgR_bin, 1);


% Find all connected components
cc = bwconncomp(imgR_bin);

% Compute the centroid of the connected components
centroids = regionprops(cc,'centroid');


matrix = [23,15;
          30,14]




% display(centroids)


% If multiple centroid coordinates have been found, chose the coordinate
% pair that are closest to the image centre
if length(matrix) > 1
    
    % Convert structure into matrix
    coors_prev = cell2mat(struct2cell(matrix));

    display(coors_prev)
    
    % Reshape matrix
    coors(:, 1) = coors_prev(1:2:end);
    coors(:, 2) = coors_prev(2:2:end);
    
    % Find the image centre coordinates
    centre = size(imgR) ./ 2;
    
    % Replicate the centre coordinates for as many centroids found
    centre = repmat(centre, length(centroids), 1);
    
    % Compute Euclidean distance between centre and centroids
    dist = (centre - coors) .^ 2;
    eucl_dist = sqrt(dist(:, 1) + dist(:, 2));
    
    % Choose the centroid corresponding to the minimum Euclidean distance
    [val, ind] = min(eucl_dist);
    centroid = coors(ind, :);
    
    % Plot centroid
    figure; imshow(img);
    hold on; plot(coors(:, 1), coors(:, 2), 'ro');
    hold on; plot(centroid(1), centroid(2), 'x');    
    
end