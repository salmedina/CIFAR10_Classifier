load('/Users/zal/CMU/Fall2015/10601/FinalProject/Dataset/cifar-10-batches-mat/data_batch_2.mat')
load('Model2.mat')
[m,n]=size(data);
y = nn_hog_classify(Model,data);
tp = sum(y==labels);
precision = tp/m;
display(sprintf('Precision: %f',precision));