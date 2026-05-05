close all;
clear all;

x = [1:50]';
A = [ones(size(x)) x];
for i = 1:9
    noise = 10*(i-1);
    y = x + noise*rand(size(x))+20;
    subplot(3,3,i);
    plot(x,y,'*');
    r = corrcoef(x,y).^2;
    text(5,105,['Noise = ',num2str(noise)])
    text(5,95,['r^2 = ',num2str(r(1,2)^2)]);
    xlabel('x');
    ylabel('y = x + 20 + noise*randn');
    axis([0 50 0 120]);
    
    m = inv(A'*A)*A'*y;
    ym = A*m;
    
    hold on;
    plot(x,ym,'r');
    text(5,85,['y = ',num2str(m(2)),'x + ',num2str(m(1))]);
end