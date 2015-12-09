function [ Y ] = classify( Model, X )
   Y=nn_hog_classify(Model,X);
end

function [gv] = GuassianValue(mean, variance, xValue)
    exponent = -(((xValue - mean).^2) / (2*variance));
    base = sqrt(2*pi*variance);
    gv = exp(exponent) / base;
end

