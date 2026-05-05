function varargout = hfsurvey(v);

load('hf_data.mat');
quad = 1;

c = v(5,:);
v = v(1:4,:);

in = inpolygon(hf_data.lon,hf_data.lat,v(:,1),v(:,2));
%figure(1); hold on;
%plot(hf_data.lon(in),hf_data.lat(in),'b.');

x = 6371*sphangle(hf_data.lon(in),hf_data.lat(in),c(1),c(2),quad);

if nargout > 0
    varargout{1} = x;
    varargout{2} = hf_data.qc(in,1);
    varargout{3} = hf_data.qu(in,2);
    return
end

if isempty(find(isnan(hf_data.qc(in,2)) == 0))
    errorbar(x,hf_data.qc(in,1),hf_data.qc(in,2),'bo');
else
    errorbar(x,hf_data.qc(in,1),hf_data.qu(in,2),'ko');
end

xlabel('Distance [km]');
ylabel('Heat Flow [mW/m2]');

return
