function [ d ] = sift_feat( image )
%% check input data type
if ~isa(image, 'single'), image = single(image); end;

I = single(rgb2gray(image)) ;
[f,d] = vl_sift(I) ;

end

