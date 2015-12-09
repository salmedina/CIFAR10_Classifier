function [Y]=nn_hog_classify(model,X)
    % NN was trained on all HoG features per image (D=496)
    hog_descriptors = extract_all_hog(X,8);
    output = feed_forward(model,hog_descriptors);
    [~,Y]=max(output,[],2);
    Y = Y-1; %Trained with values from 1 to 10
end