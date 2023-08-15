clear;
clc;

% rng default
n = 512; 
d = 2; 

% https://www.mathworks.com/help/stats/generating-quasi-random-numbers.html
%% Uniform pseudorandom numbers
figure;

subplot(1,4,1)
X = rand(n,d);
scatter(X(:,1),X(:,2),5,'b')
axis square
title('{\bf Random}')

%% Latin hypercube sequences
% Criterion: 'maximin','correlation'
subplot(1,4,2)
X = lhsdesign(n,d,'Criterion','maximin');
scatter(X(:,1),X(:,2),5,'b')
axis square
title('{\bf LHS}')

%% Sobol sequences
% https://www.mathworks.com/help/stats/sobolset.html
% sobolset
subplot(1,4,3)
p = sobolset(d, 'Skip',1e3,'Leap',1e2);
X = net(p,n);
scatter(X(:,1),X(:,2),5,'r')
axis square
title('{\bf Sobol}')


%% Halton sequences
%[1] Kocis, L., and W. J. Whiten. "Computational Investigations of...
% Low-Discrepancy Sequences." ACM Transactions on Mathematical...
% Software. Vol. 23, No. 2, 1997, pp. 266–294.
% One rule for choosing Leap values for Halton sets is to set the value to ...
% (n–1), where n is a prime number that has not been used to generate one...
% of the dimensions. For example, for a d-dimensional point set, specify...
% the (d+1)th or greater prime number for n.
% nthprime( ) requires Symbolic Math Toolbox 
subplot(1,4,4)
p = haltonset(d,'Skip',1e3,'Leap',nthprime(d+1)-1);
p = scramble(p,'RR2');
X = net(p,n);
scatter(X(:,1),X(:,2),5,'r')
axis square
title('{\bf Halton}')

% figure
% q = qrandstream(p);
% X = qrand(q,n);
% scatter(X(:,1),X(:,2),5,'b')
% axis square
% title('{\bf Halton stream}')

%%
%% Sobol sequences (standard)
% https://www.mathworks.com/help/stats/sobolset.html
% sobolset
subplot(2,1,1)
p = sobolset(d);
X = net(p,n);
scatter(X(:,1),X(:,2),5,'r')
box on
axis square
title('{\bf Sobol (MATLAB)}')


%% Halton standard
subplot(2,1,2)
p = haltonset(d);
X = net(p,n);
scatter(X(:,1),X(:,2),5,'r')
box on
axis square
title('{\bf Halton (MATLAB)}')








