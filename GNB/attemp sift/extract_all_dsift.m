function descriptors=extract_all_dsift(batch_data,win_size)
% Calculates all the dense SIFT vectors for a matrix with 
% with images described per rows
    [~, descriptors]=extract_dsift(batch_data,1,win_size); %get the 1st dSIFT descriptor
    [num_images,~]=size(batch_data); % get the number of total images (rows) in the data 
    for i=2:num_images %First image was captured above
        [~, d_i]=extract_dsift(batch_data,i,win_size); %VL_feat returns vectors in columns
        descriptors=[descriptors d_i]; %concatenate the vectors horizontally
    end
end