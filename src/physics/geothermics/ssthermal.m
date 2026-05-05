function [T,q0] = ssthermal(x,z,Ts,Tb,qs,qflag,zb,bflag,k,h,varargin);

tol = 0.0002;
if nargin == 11
    tol = varargin{1}
elseif nargin > 11 | nargin < 10
    error('ERROR (ssthermal): Incorrect number of arguments.');
end

z = z(:);
x = x(:)';
if length(qs) == 1
    qs = qs*ones(size(x));
end

nz = length(z);
nx = length(x);

if length(Tb) == 1
    Tb = Tb*ones(size(x));
elseif length(Tb) ~= nx
    error('ERROR (ssthermal): Tb length invalid.');
end

if length(Ts) == 1
    Ts = Ts*ones(size(x));
elseif length(Ts) ~= nx
    error('ERROR (ssthermal): Ts length invalid.');
end

K = zeros([nz,nx,4]);
H = zeros([nz,nx]);

% Dimensions progress [1,2,3,4] = [up,down,left,right]
K(2:nz,2:nx-1,1) = 0.5*(k(1:end,1:end-1) + k(1:end,2:end));
K(2:nz,1,1) = k(1:end,1);
K(2:nz,end,1) = k(1:end,end);
K(1:nz-1,2:nx-1,2) = K(2:nz,2:nx-1,1);
K(1:nz-1,1,2) = k(1:end,1);
K(1:nz-1,end,2) = k(1:end,end);
K(2:nz-1,2:nx,3) = 0.5*(k(1:end-1,1:end) + k(2:end,1:end));
K(2:nz-1,1:nx-1,4) = K(2:nz-1,2:nx,3);

%figure;
%subplot(3,3,5);
%imagesc(0.5*(x(1:end-1) + x(2:end)),0.5*(z(1:end-1) + z(2:end)),k);
%title('k cell');
%
%subplot(3,3,2);
%imagesc(x,z,K(:,:,1));
%title('upper node');
%
%subplot(3,3,8);
%imagesc(x,z,K(:,:,2));
%title('lower node');
%
%subplot(3,3,4);
%imagesc(x,z,K(:,:,3));
%title('left node');
%
%subplot(3,3,6);
%imagesc(x,z,K(:,:,4));
%title('right node');

H(2:nz-1,2:nx-1) = 0.25*(h(1:end-1,1:end-1) + h(1:end-1,2:end) ...
    + h(2:end,1:end-1) + h(2:end,2:end));
H(2:nz-1,1) = 0.5*(h(1:end-1,1) + h(2:end,1));
H(2:nz-1,end) = 0.5*(h(1:end-1,end) + h(2:end,end));

%figure;
%subplot(121);
%imagesc(0.5*(x(1:end-1) + x(2:end)),0.5*(z(1:end-1) + z(2:end)),h);
%title('h cell');
%
%subplot(122);
%imagesc(x,z,H);
%title('h node');

T = zeros([nz nx]);
T(1,:) = Ts;
for i = 1:nx
    T(:,i) = T1d(z,Ts(i),Tb(i),qs(i),[H(1,i); H(2:end-1,i)],K(:,i,2));

    ind = min(find(z >= zb(i)));
    T(ind:end,i) = Tb(i);
end

%save T1d.mat T
%error

figure;
subplot(221);
plot(x,K(1,:,2).*(T(2,:) - T(1,:))/(z(2) - z(1)),'r');
ylabel('Heat Flow [mW/m^{2}]');
xlabel('Distance [km]');

subplot(223);
v = linspace(min(Ts),max(Tb),10);
contour(x,z,T,v); hold on;
plot(x,zb,'k-');
title('T guess');
ylabel('Depth [km]');
xlabel('Distance [km]');
axis ij;

c = 1;
iz = 1./diff(z).^2;
ix = 1./diff(x).^2;

if bflag == 0
    zb(1:end) = z(end);
end

while 1
    Told = T;
    if qflag == 1
        T = Tsolve(Told,z,iz,ix,zb,K,H,qs);
    else
        T = Tsolve(Told,z,iz,ix,zb,K,H);
    end

    if max(abs(Told(:) - T(:))) < tol
        break;
    end

    q0 = K(1,:,2).*(T(2,:) - T(1,:))/(z(2) - z(1));

    if c == 1 | mod(c,10) == 0
        subplot(222);
        plot(x,qs,'r-');
        hold on;
        plot(x,q0,'b-');
        ylabel('Heat Flow [mW/m^{2}]');
        xlabel('Distance [km]');
        hold off;

        subplot(224);
        contour(x,z,T,v);
        hold on;
        plot(x,zb,'k-');
        title('Temperature Solution');
        ylabel('Depth [km]');
        xlabel('Distance [km]');
        axis ij;
        drawnow
        hold off;
    end
    c = c + 1;
end


return


function Tn = Tsolve(To,z,iz,ix,zb,K,H,varargin)

nx = length(ix);
nz = length(iz);

if nargin == 8
    q0 = varargin{1};
    start = 3;
    dz = sqrt(1/iz(1));
    for j = 1:nx
        To(2,j) = To(1,j) + dz/K(2,j,1)*(q0(j) - 0.5*H(2,j)*dz(1));
    end
else
    start = 2;
end

Tn = To;

for i = start:nz
    for j = 2:nx-1
        if z(i) > zb(j)
            continue;
        end
        Tn(i,j) = ( H(i,j) ...
            + To(i-1,j)*K(i,j,1)*iz(i-1) ...
            + To(i+1,j)*K(i,j,2)*iz(i) ...
            + To(i,j-1)*K(i,j,3)*ix(j-1) ...
            + To(i,j+1)*K(i,j,4)*ix(j) ) ...
            / ( K(i,j,1)*iz(i-1) + K(i,j,2)*iz(i) + K(i,j,3)*ix(j-1) + K(i,j,4)*ix(j) );
    end
end

% apply insulating boundary conditions, i.e. dT/dx|vertical boundary = 0
Tn(:,1) = Tn(:,2);
Tn(:,end) = Tn(:,end-1);

return
