function [hog_descriptors] = extract_all_hog(batch_data, win_size)
    [m,n]=size(batch_data);
    hog_descriptors = vl_hog(im2single(rgb2gray(reshape(batch_data(1,:),32,32,3))), win_size);
    hog_descriptors = hog_descriptors(:);
    for i=2:m
        desc_i = vl_hog(im2single(rgb2gray(reshape(batch_data(i,:),32,32,3))), win_size);
        hog_descriptors = [hog_descriptors, desc_i(:)];
    end
    hog_descriptors=hog_descriptors';
end
