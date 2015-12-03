function bow_data=transform_data(data_vw, span, num_bins)
    [size_data,~]=size(data_vw);
    num_images=fix(size_data/span);
    bow_data=zeros(num_images,num_bins);
    cur_pos=1;
    for i=1:64:size_data-63
        bow_data(cur_pos,:)=histcounts(data_vw(i:i+63),num_bins);
        cur_pos=cur_pos+1;
    end
end