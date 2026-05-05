%close all;
clear all;

addpath '../'
%addpath '/home/dhasterok/devel/ssthermal/';

dx = 0.2;
x = [0:dx:240];

% determine depth to sediment-basement interface (sediment thickness)
d = load('louden97_depths.dat');
ind = find(isnan(d(:,1))); 

% depth of seafloor
zsf = interp1(d(1:ind-1,1),d(1:ind-1,2),x);
% depth of igneous basement
zig = interp1(d(ind+1:end,1),d(ind+1:end,2),x);
h = zig - zsf;

% discritize sediment thickness to dz
dz = 0.05;
dig = round(h/dz)*dz;
z = [0:dz:max(dig)+dz]';

nc = length(x);
nr = length(z);

Ts = 2.5;
Tb = 40 + (65 - 40)/240*x;
%Tb(x < 102) = 40;
%Tb(x >= 102) = 55;


k0 = 2.25;
A0 = 1.5;

lambda = 0.43;
ks = k0*(1 - exp(-lambda*(z(1:end-1)+dz/2)));
K = repmat(ks,[1 nc-1]);
%H = zeros([nr-1 nc-1]);
H = repmat(A0*(1 - exp(-lambda*(z(1:end-1)+dz/2))),[1 nc-1]);

v = [-14 40;
    -14 41.5;
    -10 41.5;
    -10 40;
    -13.3 41]; % Iberian Abyssal Plain, Louden et al. [EPSL, 1997]
[xq,qo,qe] = hfsurvey(v);

qs = interp1(xq,qo,x);
qs(isnan(qs)) = median(qs(~isnan(qs)));

%[T,q2d] = ssthermal(x,z,Ts,Tb,qs,0,dig,1,K,H,5e-3);
[T,q2d] = ssthermal(x,z,Ts,Tb,30,0,dig,1,K,H,5e-3);

q1d = (Tb - Ts)*k0*lambda./( log(1 - k0*exp(lambda*h)) - log(1 - k0) );

t = 126;
vs = h/t;
ks = k0 - exp(-lambda*h);
%%kappa = ??
%X = vs*sqrt(t)/sqrt(kappa);
%cs = 1 - (1 - 2*X.^2)*erfc(X) - 2/pi*X*exp(-X.^2)


figure;
subplot(211);
plot(x,q1d,'b-'); hold on;
plot(x,q2d,'r-'); hold on;
errorbar(xq,qo,qe,'ko');
xlabel('Distance [km]');
ylabel('Heat Flow [mW/m2]');
axis([0 240 20 100]);

subplot(212);
imagesc(x,z,T);
hold on;
plot(x,dig,'k-');
xlabel('Distance');
ylabel('Depth [km]');
set(gca,'DataAspectRatio',[10 1 1]);

