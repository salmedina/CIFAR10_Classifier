function output_labels = convert_labels(labels, k)
    [m,n]=size(labels);
    output_labels=zeros(m,k);
    for t=1:k
        output_labels(:,t)=labels==t;
    end
end