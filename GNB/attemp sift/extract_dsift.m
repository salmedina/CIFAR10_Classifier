function [f,d]=extract_dsift(batch_data, index_num, winSize)
% Extract the dense SIFT vectors per image

    % 1. reshape the image from vector into a 32x32 with 3 channels
    % 2. transform into grayscale space
    % 3. set as SINGLE for VL_Feat
    % 4. set the window size for dSIFT extraction
    % 5. return the frames and descriptor vectors of the image
    [f,d]=vl_dsift(im2single(rgb2gray(reshape(batch_data(index_num,:),32,32,3))),'size',winSize);
end