function [S, vopts] = LSIR(X, Y, d, s, opts)

% Input: 
% X: p X n input data matrix
% Y: response variable
% d: number of LSIR directions
% s: regularization parameter
% opts: structrue containing parameters
%       [pType]: Type of problem: 'c' for classification
%                               'r' for regression
%       [H]: number of slices, defalt: 10 for 'r' and class number for 'c'
%       [numNN]: number of nearest neighbors
%
% Output:
% S:   a structure containing
%      [edrs] : edr directions
%      [Xmean]: mean of the samples
%      [Xv]: centered LSIR variates for input data X
%      [Cov]: Covariance matrix of inverse regression
%      [Sigma]: Covariance matrix of X
% vopts: structure of parameters used
%
% last modified: 9/16/2008 (documentation)

[dim n] = size(X);
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
        Xi = X(:,ind);
        for j=1:ni
            dist2j = sum((Xi-repmat(Xi(:,j),1,ni)).^2);
            [dvals, dI] = sort(dist2j);
            J(ind(dI(1:numNNi)),ind(j)) = 1;
        end
    end
end

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
            ind =YI((opts.H-1)*Hn+1:n);
        end
        ni = length(ind);
        numNNi = min(ni, opts.numNN);
        Xi = X(:,ind);
        for j=1:ni
            dist2j = sum((Xi-repmat(Xi(:,j),1,ni)).^2);
            [dvals, dI] = sort(dist2j);
            J(ind(dI(1:numNNi)),ind(j)) = 1/numNNi;
        end
    end
end

J = J'*J;

Xmean = mean(X, 2);
cX = X - repmat(Xmean, 1, n);
eigopts.issym = true;
eigopts.isreal = true;
eigopts.disp = 0;
Sigma = cX*J*cX';
Sigma = .5*(Sigma + Sigma');
Cov = cX*cX';
Cov = .5*(Cov + Cov');
[B L] = eig(Sigma, Cov + s*eye(dim));
[Lvals LI] = sort(diag(L),'descend');
B = B(:,LI(1:d));
for i = 1:d
    B(:,i) = B(:,i)/norm(B(:,i));
    [maxbi maxi] = max(abs(B(:,i)));
    B(:,i) = B(:,i)*sign(B(maxi,i));
end

S.edrs = B;
S.Xmean = Xmean;
S.Xv = B'*cX;
S.Cov = Cov;
S.Sigma = Sigma;

vopts = opts;
vopts.d = d;
vopts.s = s;
vopts.J = J;



