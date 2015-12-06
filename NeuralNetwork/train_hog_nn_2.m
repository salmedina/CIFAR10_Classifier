load('NN_train_setup.mat');
display('Environment loaded');
display('Begin NN training');
[hog_model hog_cc_train hog_output_train] = train_mlp(hog_descriptors, Y_10, [200 100], 2000, 0.1, 0.5);
display('Finished training');
save('trained_hog_nn_2.m');
display('Done');