function descriptors=extract_all_dsift(data,win_size)
    [~, descriptors]=extract_dsift(data,1,win_size);
    [num_images,~]=size(data);
    for i=2:num_images
        [~, d_i]=extract_dsift(data,i,win_size);
        descriptors=[descriptors d_i];
    end
end