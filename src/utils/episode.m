% function to solve ODE for each magmatic flare-up and deformation event
% and generate arrays of crustal thickness (thickened part of crust),
% thickening rate, erosion rate
% this function will be called in the main code MDE_model.m
% episode.m and MDE_model.m should be placed under the same directory
% Wenrong Cao, 2015-8-14. Comments improved in 2016-5
%------------------
% Inputs explained
%------------------
% t1, t2: time frame for solving ODE, t1 and t2 are the start and end time
% of a magmatic-deformational event [Myr, forward time]
% H0: thickness of crust at time=t1 [km]
% deltaH0: thickned part of crust at time=t1. Note, this is not
% the thickness of crust [km]
% Tau: ersoion response time [Myr]
% t_m_s: magmatic flare start time [Myr]
% t_m_e: magmatic flare end time [Myr]
% t_d_s: deformation start time [Myr]
% t_d_e: deformation end time [Myr]
% t_r_s: root growth start time [Myr]
% t_r_e: root growth end time [Myr]
% TM: magmatic thickening rato (same as beta in text)
% TT: total thickening ratio (magmatic and deformation combined)
% gamma: mass-of-root to mass-of-melt ratio
% t_r_f: root foundering time [Myr]
%--------------------
% Outputs explained
%--------------------
% time: time array [Myr, forward time]
% H: thickened part of crust [km]
% R: root thickness [km]
% h: elevation [km]
% E_dot: erosion rate [km/Myr]
% M_dot: magmatic thickening rate [km/Myr]
% T_dot: total thickening rate [km/Myr]
% R_dot: root growth rate [km/Myr]
function [time,H,R,h,E_dot,M_dot,T_dot,R_dot] = ...
episode (t1,t2, H0, deltaH0, Tau,t_m_s, t_m_e,...
t_d_s, t_d_e, t_r_s, t_r_e, TM, TT, gamma,t_r_f)
[t,y] = ode45(@crust, [t1 t2], deltaH0);
% use Matlab ODE solver to solve thickend part of crust(y) vs time (t)
% define time frame for solving ODE (t1 and t2) and initial
% value (y value when time=t1)
function dydt = crust(t,y)
rhoc = 2.8; % density of crust (1000 kg/m^3)
rhom = 3.3; % density of mantle (1000 kg/m^3)
rhor = 3.5; % denstiy of root (1000 kg/m^3)
M_duration = t_m_e-t_m_s ; % calculate duration of fare-up (Myr)
D_duration = t_d_e-t_d_s ; % calculate duration of deformation (Myr)
R_duration = t_r_e-t_r_s ; % calculate duration of root formation (Myr)
TR = rhoc/rhor*TM*H0*gamma; % final thickness of root
% set magmatic thickening rate
% paramters in normal distribution
mu1 = t_m_s+M_duration/2 ; % start time
sigma1 = M_duration/6 ; % distribution in +/-3 sigma range
M_dot = TM.*normpdf (t, mu1, sigma1);
% relative magmatic thickening rate (1/Myr)
% set total thickening rate
% paramters in normal distribution
mu2 = t_d_s +M_duration/2; % start time
sigma2 = M_duration/6 ; % distribution in +/-3 sigma range
T_dot = TT.*normpdf (t, mu2, sigma2);
% relative total thickening rate (1/Myr)
% set root growth rate
mu3 = t_r_s +R_duration/2; % start time
sigma3 = R_duration/6 ; % distribution in +/-3 sigma range
R_dot = TR.*normpdf (t, mu3, sigma3); % root growth rate (km/Myr)
R = TR.*normcdf (t, mu3, sigma3); % root thickness through time (km)
ind = find (t > t_r_f) ; % define root foundering time
R(ind) = 0; % root founderd, root thickness=0
%----------------------
% Set dHdt expression
% ---------------------
% calculate elevation
% isostasy equation Eq.9 in text
h = y.*(1-rhoc/rhom)-R.*(rhor/rhom-1); %(km)
% calculate erosion rate
% elevation-erosion model Eq.6 in text
E_dot = h./Tau; % (km/Myr)
if E_dot <=0 % if erosion rate value<=0 then set erosion rate=0
E_dot =0;
end
% the expression of ODE
% mass balance equation Eq.8 in text
dydt = H0.*(T_dot)-E_dot;
dydt = dydt';
end % this is the end of dydt=@crust(t,y)
time = t; % transfer to output
H = y; % transfer to output
% after ODE y(t) and t is calculated, y(t) is the thickness
% calculate arrays again for episode.m function outputs
M_dot = TM.*normpdf (t, mu1, sigma1);
% relative magmatic thickening rate (1/Myr)
M =(1+ TM.*normcdf(t,mu1, sigma1)); % intergration of M_dot
T_dot = TT.*normpdf (t, mu2, sigma2);
% relative tectonic thickening rate (1/Myr)
T =(1+ TT.*normcdf(t,mu2, sigma2)); % intergration of T_dot
R_dot = TR.*normpdf (t, mu3, sigma3); % root growth rate
R = TR.*normcdf (t, mu3, sigma3); % root thickness through time
ind = find (time > t_r_f) ; % find root foundering time
R(ind) = 0; % root foundered, set root thickness=0
% calculate elevation array again for episode.m function outputs
h = y.*(1-rhoc/rhom)-R.*(rhor/rhom-1); %(km)
% calculate erosion rate array again for episode.m function outputs
E_dot = h./Tau; % (km/Myr)
end