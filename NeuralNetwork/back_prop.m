function [model] = back_prop(model, input, target)
    % stores all the activations from every neuron from every layer
    activations = cell(length(model.weights)+1,1);
    activations{1} = input;
    
    % Calc all neuron activations
    for i = 1:length(model.weights)
        % activations{i} is a row vector
        % model.weights{i} is a matrix of weights
        % the output of that product is a row vector of length equal to the
        % number of neurons in the next layer
        temp = activations{i} * model.weights{i} + model.biases{i}; 
        activations{i+1} = 1./(1+exp(-(temp))); % Calc the probability
    end

    % variable for holding the errors at each level
    errors = cell(length(model.weights),1);
    
    % this is the actual backprop implementation
    run_error = (target - activations{end}); %keeps track of the error
    for i = length(model.weights):-1:1
        errors{i} = activations{i+1} .* (1-activations{i+1}) .* (run_error);
        run_error = errors{i} * model.weights{i}';
    end
    
    % this code updates the weights and biases
    for i = 1:length(model.weights)
        % update weights based on the learning rate, the input activation
        % and the error
        model.weights{i} = model.weights{i} + model.learning_rate * activations{i}' * errors{i};
        % update the neuron biases as well
        model.biases{i} = model.biases{i} + model.learning_rate * errors{i};
        % it takes a while to figure out all the matrix operations, but
        % once it's done it's nice.
    end
end