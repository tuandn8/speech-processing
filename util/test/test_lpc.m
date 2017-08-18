load('x.mat')
x = x(:);
[m,n] = size(x)
X = fft(x,2^nextpow2(2*size(x,1)-1));
R = ifft(abs(X).^2);
R = R./m; % Biased autocorrelation estimate

a = lpc(x, 13)
a

%a = lpc(x, 13)
