function [max_val, max_idx]=softmax(x)
% SOFTMAX(x) obtains the index of the largest element
% after applying the softmax function
soft_x = exp(x)/sum(exp(x))
[max_val, max_idx] = max(soft_x)

