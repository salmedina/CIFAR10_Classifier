function [output] = feed_forward(model, input)
    % this function calculates the output of the model, and compares it to the
    % target output
     % this function calculates the output of the model, and compares it to the
    % target output
    ntest = size(input,1);
    nOutLayer = 10;
    
    % this variable is for the final output of the neural net
    output = zeros(ntest, nOutLayer);
    for i = 1:ntest
        temp = input(i,:); % output at each layer, gets updated
        for j = 1:length(model.weights)
            temp = temp * model.weights{j} + model.biases{j}; % calculate the output
            temp = 1./(1+exp(-temp)); % squashit
        end
        output(i,:) = temp; % keep only the last output value
    end
end