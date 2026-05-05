function T = T1d(z,Ts,Tb,qs,H,k);

dz = diff(z);
q = qs - cumsum(H.*dz);
q = [qs; q];

n = length(z);

T = zeros([n 1]);
T(1) = Ts;
flag = 0;
for i = 1:n-1
    if flag == 1
        T(i+1) = Tb;
    end
    T(i+1) = T(i) + q(i)/k(i)*dz(i) - 0.5*H(i)/k(i)*dz(i)^2; 
    if T(i+1) >= Tb
        T(i+1) = Tb;
        flag = 1;
    end
end
T = T(:);

return
