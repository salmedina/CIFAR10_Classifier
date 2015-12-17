load('NN_train_setup.m');
display('Environment loaded');
display('Begin NN [50]');
[hog_model hog_cc_train hog_output_train] = train_mlp(hog_descriptors, Y_10, [50], 700, 0.1, 0.5);
display('Finished training');
save('trained_hog_nn_50.mat');

display('Begin NN [100]');
[hog_model hog_cc_train hog_output_train] = train_mlp(hog_descriptors, Y_10, [100], 700, 0.1, 0.5);
display('Finished training');
save('trained_hog_nn_100.mat');

display('Begin NN [200]');
[hog_model hog_cc_train hog_output_train] = train_mlp(hog_descriptors, Y_10, [200], 700, 0.1, 0.5);
display('Finished training');
save('trained_hog_nn_200.mat');

display('Done');