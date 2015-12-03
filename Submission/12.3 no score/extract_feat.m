function [feat] = extract_feat(image)
%% check input data type
if ~isa(image, 'single'), image = single(image); end;

%% extract HOG 
cellSize = 8;
hog = vl_hog(image, cellSize, 'verbose');
%imhog = vl_hog('render', hog, 'verbose');
%clf; imagesc(imhog); colormap gray;

%% feature - vectorized HOG descriptor
feat = hog(:);

end
