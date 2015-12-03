function descriptors=extract_all_dsift(batch_data,win_size)
    [~, descriptors]=extract_dsift(batch_data,1,win_size);
    [num_images,~]=size(batch_data);
    for i=2:num_images
        [~, d_i]=extract_dsift(batch_data,i,win_size);
        descriptors=[descriptors d_i];
    end
end