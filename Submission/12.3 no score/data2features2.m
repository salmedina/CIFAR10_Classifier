function [ features ] = data2features2( data )
% extract features from data
    N = size(data, 1); % total number of samples
    F = cell(N, 1); % to get the feature matrix
    for n = 1 : N
        item = reshape(data(n, :), 32, 32, 3); % the format of image is given
        %feat = extract_feat(im2single(item)); % get the feature of this sinlge image
        %f1 = extract_feat(im2single(item));
        
        red = item(:,:,1);
        red(red == 0) = 1;
        red = double(red);
        green = item(:,:,2);
        green(green == 0) = 1;
        green = double(green);
        blue = item(:,:,3);
        blue(blue == 0) = 1;
        blue = double(blue);
        
        n_red = red ./ (red + green + blue);
        n_green = green ./ (red + green + blue);
        n_blue = blue ./ (red + green + blue);
        
        n_color = cat(3, n_red, n_green, n_blue);
        
        feat = extract_feat(im2single(n_color)); % get the feature of this sinlge image
        
        % get the right dimension
        if size(feat, 2) == 1
            feat = feat';
        end       
        
        F{n} = feat;
        %if size(f1, 2) == 1
        %    f1 = f1';
        %end   
        %feature = [f1, feat];
        %F{n} = feature;
    end   
    
    features = cell2mat(F); % get the NxF feature matrix, here F stand for the number of features
end

