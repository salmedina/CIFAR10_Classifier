function [model cc output] = train_mlp(input, target, hidden, epochs, learning_rate, momentum)
    % Trains through backprop with momentum
    % initialize the output
    model = [];
    model.learning_rate = learning_rate;
    model.momentum = momentum; % for some heavy ball action

    % characterize the input and output
    [ntrain nInLayer] = size(input);
    [jnk nOutLayer] = size(target);

    % keep track of how many neurons are in each layer
    nNeurons = [nInLayer hidden nOutLayer];
    nNeurons(nNeurons == 0) = []; % remove 0 layers, to allow putting a zero for no hidden layers

    % there are one fewer sets of weights and biases as total layers
    nTransitions = length(nNeurons)-1;

    for i = 1:nTransitions % initialize the weights between layers, and the biases (past the first layer)
        model.weights{i} = randn(nNeurons(i),nNeurons(i+1)); 
        % the weight matrix has X rows, where X is the number of input
        % neurons to the layer, and Y columns, where Y is the number of
        % output neurons.  multiplication of the input with the weight
        % matrix transforms the dimensionality of the input to that of the
        % output.  Initialization is done here randomly.
        model.biases{i} = randn(1,nNeurons(i+1));
        % biases are random as well
        model.lastdelta{i} = 0;
    end
    
    for cur_epoch = 1:epochs % repeat the whole training set over and over
        if mod(cur_epoch,10)==0
            display(sprintf('Epoch: %d',cur_epoch));
        end
        order = randperm(ntrain);  % shuffle the training samples
        for j = 1:ntrain
            model = back_prop(model, input(order(j),:), target(order(j),:));
        end
    end
    % test the performance on the training set after all is trained.
    [output cc] = test_mlp(model, input, target);
end
