% FEED FORWARD
%activation_func=@(z) tanh(z)
activation_func=@(z) 1.0 ./(1.0+exp(-z));

input_layer_size=2;
hidden_layer_size=3;
output_layer_size=1;

W1=randn(input_layer_size, hidden_layer_size);
W2=randn(hidden_layer_size, output_layer_size);

Z2=X*W1;
A2=activation_func(Z2);

Z3=A2*W2;

Y_out=activation_func(Z3);
