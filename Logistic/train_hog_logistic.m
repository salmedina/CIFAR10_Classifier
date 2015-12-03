function [theta]=train_hog_logistic(data,y)
    y=y+1;
    hog_descriptors=double(extract_all_hog(data,8));
    [theta,J]=train_multinomial_logistic(hog_descriptors,y,10,0.1,10000);
    Model1=cell(1);
    Model1{1}=theta;
    save('Model1.mat','Model1');
end