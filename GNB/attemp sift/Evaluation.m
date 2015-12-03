function [ Precision ] = Evaluation( labels, Y )
%UNTITLED11 Summary of this function goes here
%   Detailed explanation goes here
    N = size(labels, 1);
    count = 0;
    
    for i = 1:N
        if labels(i,1) == Y(i,1)
            count = count + 1;
        end 
    end
    Precision = count / N;
end

