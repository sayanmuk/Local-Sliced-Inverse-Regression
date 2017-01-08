function ytest = knn(lx,ly,xtest,k)

% this function returns the labels by 1-nn methods

[dim n] =size(lx);

ytest = zeros(1,size(xtest,2));
for i = 1:size(xtest,2)
    dist = zeros(1,n);
    for j = 1:n
        dist(j) = norm(xtest(:,i)-lx(:,j));
    end
    
    [distvals distI] = sort(dist);
    temp = ly(distI(1:k)); % contains labels of k neaerest neighbor points

    num = zeros(k,1);
    for j=1:k
        num(j) = length(find(temp==temp(j)));
    end
    [maxnum idx] = max(num);
    ytest(i) = temp(idx);
    
end
