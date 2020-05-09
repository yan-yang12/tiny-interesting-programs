function pred = predict_svm(x, type, vec, alpha, bias, r)
[~, sv] = size(vec);
h = 0;
if type == "linear" 
    for i = (1:sv)
        h = h + alpha(i) * vec(end,i) * (x * vec(1:end-1,i));
    end
elseif type == "polynomial"
    for i = (1:sv)
        h = h + alpha(i) * vec(end,i) * (x * vec(1:end-1,i) + 1) .^ r;
    end
elseif type == "rbf"
    for i = (1:sv)
        h = h + alpha(i) * vec(end,i) * exp(-r .* vecnorm(vec(1:end-1,i) - x', 2).^2);
    end
    h=h';
end
pred = sign(h + bias);
end