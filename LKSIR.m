function [SIR, vopts] = LKSIR(X, K, Y, d, s, opts)

% Input: 
% K: n X n kernel matrix; K(i,j)=K(x_i,x_j)
% Y: response variable
% d: number of SIR directions
% s: tuning parameter
% opts: structrue containing parameters
%       pType: Type of problem: 'c' for classification
%                               'r' for regression
%       H: number of slices, defalt: 10 for 'r' and class number for 'c'
%       NeighborMat: Nearest Neighbor Matrix, 
%       numNN: number of nearest neighbors
%       
%
% Output:
% SIR: a structure containing
%      C : row vector of length n contains
%          coefficients for compute sir variates for new data
%      b : center parameter
%  for a given new data, the SIR variate is 
%          C*(K[x_i, x]-mean(K[x_i,x])) + b;
% 
% vopts: structure of parameters used
%
% last modified: 1/8/2008

n=length(K(:,1));
if nargin<5
    opts=[];
end

if ~isfield(opts, 'pType')
    if (~isnumeric(Y))|| (length(unique(Y))<5)
        opts.pType = 'c';
    else
        opts.pType = 'r';
    end
end

J=zeros(n,n);

if opts.pType=='c'
    labels = unique(Y);
    opts.H = length(labels);
    for i = 1:opts.H
        ind = find(Y==labels(i));
        ni = length(ind);
        numNNi = min(ni, opts.numNN);
%         numNNi = ceil(ni*.5);
        Xi = X(:,ind);
        for j=1:ni
            dist2j = sum((Xi-repmat(Xi(:,j),1,ni)).^2);
            [dvals, dI] = sort(dist2j);
            J(ind(dI(1:numNNi)),ind(j)) = 1;
        end
    end
end


% compute J for r problem

if opts.pType=='r'
    if ~isfield(opts, 'H')
        opts.H = 10;
    end
    [Yvals, YI] = sort(Y);
    Hn = round(n/opts.H);
    for i = 1:opts.H
        if i<opts.H
            ind = YI((i-1)*Hn+1:i*Hn);
        else
            ind = YI((opts.H-1)*Hn+1:n);
        end
        Xi = X(:,ind);
        ni = length(ind);
        numNNi = min(ni, opts.numNN);
        for j=1:ni
            dist2j = sum((Xi-repmat(Xi(:,j),1,ni)).^2);
            [dvals, dI] = sort(dist2j);
            J(ind(dI(1:numNNi)),ind(j)) = 1;
        end
    end
end

J = J'*J;

Kmean = mean(K);
cK = K - repmat(Kmean,n,1) - repmat(mean(K,2),1,n) + mean(Kmean);
cK = .5*(cK+cK);
[U D V] = svd(.5*(cK+cK'));
D = diag(D);
Dvals = sort(D,'descend');
if nargin<4 || isempty(s)
    s = .1*Dvals(d);
    if s==0
        warning('parameter d is too large')
    end
end
ind  = find(D>1.0e-10*Dvals(1));
D = D(ind); 
U = U(:,ind);
Ds = diag(D+s);
D = diag(sqrt(D));
[V L] = eig(D*U'*J*U*D, Ds);
[Lvals LI] = sort(diag(L),1,'descend');
V = V(:,LI(1:d));
C = U*Ds^(-1/2)*V;

% Gamma = cK*J*cK;
% Gamma = .5*(Gamma+Gamma');
% Sigma = cK*cK;
% Sigma = .5*(Sigma+Sigma') + s*eye(n);
% % eigopts.disp = 0;
% % [C L] = eigs(Gamma, Sigma+s*eye(n), d, 'lm', eigopts)
% 
% [C L] = eig(Gamma, Sigma);
% [Lvals LI] = sort(diag(L), 'descend');
% C = C(:, LI(1:d));

SIR.C = C';
SIR.b = C'*(mean(Kmean)-Kmean)';
SIR.X = C'*cK;

vopts = opts;
vopts.d = d;
vopts.s = s;