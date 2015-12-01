function [f,d]=extract_dsift(batch_data, index_num, winSize)
    [f,d]=vl_dsift(im2single(rgb2gray(reshape(batch_data(index_num,:),32,32,3))),'size',winSize);
end