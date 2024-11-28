%% Clearing Termina
clc,clearvars

img = imread('I2head_5_left.png')
grayScale = rgb2gray(img)

r = img(:,:,1)

gaussian = imgaussfilt(r,1)


subplot(3,1,1)
imshow(grayScale)
subplot(3,1,2)
imhist(grayScale)
subplot(3,1,3)
imshow(gaussian)