load('NN_train_setup.m');
display('Environment loaded');
display('Begin NN training');
[hog_model hog_cc_train hog_output_train] = train_mlp(hog_descriptors, Y_10, [100 100], 1000, 0.1, 0.5);
display('Finished training');
save('trained_hog_nn_100_100.m');
display('Done');