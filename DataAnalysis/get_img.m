function img=get_img(batch_data, index_num)
    img=reshape(batch_data(index_num,:),32,32,3);
end