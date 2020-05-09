function [type, r, vec, coeff, bias] = svm(x, y, C, type, r)
%SVM a simple SVM implementation
%input dimension x(n,d) y(n,1)
%preparation of data
[m, ~] = size(x);
x = x';
y = y';

%quadprog parameters
f = -ones(m,1);
Aeq = y;
beq = 0;
lb=zeros(m,1);
ub=C*ones(m,1);
options=optimset('MaxIter', 5000, 'Display', 'off');

if (lower(type) == "linear")
    %compute H
    H = (y'*y) .* (x'*x);
    
    alpha = quadprog(H, f, [], [], Aeq, beq, lb,ub,[],options)';
    
    %margin of error is 1e-6
    alpha(alpha <= C*1e-6) = 0;
    alpha(alpha >= (1-1e-6)*C) = C;
     
    S = find(alpha>0 & alpha<=C);
    S1 = find(alpha>0 & alpha<C);
    
    %return alpha
    coeff = alpha(S);
    
    %return support vectors, the last row being all y's
    counter = 1;
    for i = S
        vec(:,counter) = x(:,i);
        counter = counter + 1;
    end
    vec(end+1, :) = y(S);
    
    %return bias
    bias = 0;
    for i = S1 
        temp = 0;
        for j = S
            temp = temp + alpha(j) * y(j) * (x(:,i)' * x(:,j));
        end
        bias = bias + y(i) - temp;
    end
    bias = bias / length(S1);
    
    %-----------------------Ploynomial case--------------------------------
elseif (lower(type) == "polynomial")
    %compute H
     H = zeros(m,m);
     for i=1:m
         for j=i:m
             H(i,j) = y(i) * y(j) * (x(:,i)'*x(:,j) + 1) .^ r;
             H(j,i) = H(i,j);
         end
     end
    
    alpha = quadprog(H, f, [], [], Aeq, beq, lb,ub,[],options)';
    
    alpha(alpha <= C*1e-6) = 0;
    alpha(alpha >= (1-1e-6)*C) = C;
     
    S = find(alpha>0 & alpha<=C);
    S1 = find(alpha>0 & alpha<C);

    %return alpha
    coeff = alpha(S);

    %return all support vectors
    counter = 1;
    for i = S
        vec(:,counter) = x(:,i);
        counter = counter + 1;
    end
    vec(end+1, :) = y(S);
    
    %return bias
    bias = 0;
    for i = S1 
        temp = 0;
        for j = S
            temp = temp + alpha(j) * y(j) * (x(:,i)' * x(:,j) + 1) .^r;
        end
        bias = bias + y(i) - temp;
    end
    bias = bias / length(S1);
    
    %------------------------------RBF-------------------------------------
elseif lower(type) == "rbf"
    %compute H
    H = zeros(m,m);
    for i=1:m
        for j=i:m
            H(i,j) = y(i) * y(j) * exp(-r * norm(x(:,i)-x(:,j),2).^2);
            H(j,i) = H(i,j);
        end
    end
    
    alpha = quadprog(H, f, [], [], Aeq, beq, lb,ub,[],options)';
    
    alpha(alpha <= C*1e-6) = 0;
    alpha(alpha >= (1-1e-6)*C) = C;
     
    S = find(alpha>0 & alpha<=C);
    S1 = find(alpha>0 & alpha<C);
    
    %return alpha
    coeff = alpha(S);

    %return all support vectors
    counter = 1;
    for i = S
        vec(:,counter) = x(:,i);
        counter = counter + 1;
    end
    vec(end+1, :) = y(S);
    
    %return bias
    bias = 0;
    for i = S1 
        temp = 0;
        for j = S
            temp = temp + alpha(j) * y(j) * exp(-r * vecnorm(x(:,i) - x(:,j),2).^2);
        end
        bias = bias + y(i) - temp;
    end
    bias = bias / length(S1);
end