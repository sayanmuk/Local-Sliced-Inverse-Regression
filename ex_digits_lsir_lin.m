clear

load all_dig

%% set parameters
T =100; % set repeating times

errlin = zeros(1,T);
errlineach = zeros(10, T);
plin = zeros(1,T);


d = 20;
opts.pType = 'c';
opts.numNN = 30;
    
%% generate training and test data 

xt = test_dig';

for it =1:T

%     tic
%     it
    n = 100;
    N = n*10;
    x = zeros(N,784);
    y = zeros(N,1);
    for i = 1:10
        j = i-1;
        idxj = find(lab == j);
        lj = length(idxj);
        sj = randperm(lj);
        sj = sj(1:n);
        sj = idxj(sj);
        x((j*n+1):(j+1)*n,:) = dig(sj, :);
        y(j*n+1:(j+1)*n) = j;
    end

    x = x';

    %%  linear kernel case
    K = x'*x;

    Kxxt = x'*xt;

%% cross validate the parameter 

    s = 10.^([1:10]);
    errlintemp = zeros(1,10);
    for cv = 1:10  % 10-fold cv
        cvsetidx = zeros(1, n);
        for j = 1:10
            cvsetidx(((j-1)*n/10)+[1:(n/10)]) = (j-1)*n + (cv-1)*n/10 + [1:(n/10)];
        end

        trsetidx = 1:N;
        trsetidx(cvsetidx) = [];
        
        xcvtr = x(:, trsetidx);
        
        Kcvtr = K(trsetidx, trsetidx);
        Kcvtest = K(trsetidx, cvsetidx);

        ycvtr = y;
        ycvtr(cvsetidx) = [];
        ycvtest = y(cvsetidx);
        
        for i=1:10

        sircv =  LKSIR(xcvtr, Kcvtr, ycvtr, d, s(i), opts);

        xcvtest = sircv.C*(Kcvtest - repmat(mean(Kcvtest, 1), length(trsetidx), 1)) + repmat(sircv.b, 1, length(cvsetidx));
        predcv = knn(sircv.X, ycvtr, xcvtest, 5);
        errlintemp(i) = errlintemp(i) + length(find(predcv~= ycvtest'));
        end
    end

    [errcvmin pidx] = min(errlintemp);
    plin(it) = s(pidx);

%% training and test
    
    sir = LKSIR(x,K,y,d, plin(it) ,opts);

    newxt = zeros(d, size(xt,2));
    for i=1:size(xt,2)
        newxt(:,i) = sir.C*(Kxxt(:,i) - mean(Kxxt(:,i))) +sir.b;
    end

    pred = knn(sir.X, y, newxt, 5);
    errlin(it) = length(find(pred ~= test_lab'))/10000

    for c = 1:10
        idxc = find(test_lab == c-1);
        predc = pred(idxc);
        errlineach(c, it) = length(find(predc~=c-1))/length(idxc);
    end
    
%   toc

end

%% show result

% mean(errlin)
% [mean(errlineach,2) std(errlineach,0, 2)]

eval(['save digits_lsir_100_d' num2str(d) ' errlin errlineach plin'])

