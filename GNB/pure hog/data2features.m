 function [ features ] = data2features( data )
% extract features from data
    N = size(data, 1); % total number of samples
    F = cell(N, 1); % to get the feature matrix
    for n = 1 : N
        item = reshape(data(n, :), 32, 32, 3); % the format of image is given
        feat = extract_feat(im2single(item)); % get the feature of this sinlge image
       
        % get the right dimension
        if size(feat, 2) == 1
            feat = feat';
        end        
        F{n} = feat;
    end   
    
    features = cell2mat(F); % get the NxF feature matrix, here F stand for the number of features
end

