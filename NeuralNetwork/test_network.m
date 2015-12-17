% TEST the Neural net with other batches

load('/Users/zal/CMU/Fall2015/10601/FinalProject/Dataset/cifar-10-batches-mat/test_batch.mat')
[m,n] = size(data);
hog_descriptors = extract_all_hog(data,8);
binary_labels = convert_labels(labels+1,10);
[nn_output nn_cc] = test_mlp(hog_model, hog_descriptors, binary_labels);
[prob,nn_y]=max(nn_output,[],2);
nn_y = nn_y-1;
tp = sum(nn_y==labels);
precision = tp/m;
display(sprintf('Precision: %f',precision));