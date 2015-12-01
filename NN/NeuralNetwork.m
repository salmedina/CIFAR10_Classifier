classdef NeuralNetwork
    properties
        NN_input_layer_size=2;
        NN_hidden_layer_size=3;
        NN_output_layer_size=1;
        NN_W1=randn(NN_input_layer_size, NN_hidden_layer_size);
        NN_W2=randn(NN_hidden_layer_size, NN_output_layer_size);
    end
    methods
        function obj=NeuralNetwork(inputSize, outputSize, hiddenSize)
            
        end
        function Y=FeedForward(X)
            activation_func=@(z) 1.0 ./(1.0+exp(-z));
            Z2=X*NN_W1;
            A2=activation_func(Z2);
            Z3=A2*NN_W2;
            Y=activation_func(Z3);
        end
    end
end