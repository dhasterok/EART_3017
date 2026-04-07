function gl = glatcorr(phi)

phi = phi*pi/180;

gl = 9.78031846*(1 + 0.0053024*sin(phi).^2 - 5.80e-6*sin(2*phi).^2);

return