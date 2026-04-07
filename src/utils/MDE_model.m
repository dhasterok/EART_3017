% Magmatism-Deformation-Erosion coupling model (MDE model) to calculate
% temporal evolutions of crustal thickness, elevation, erosion rate with
% parameterized magmatism and tectonism.
% This code uses episode.m and total_thickness.m functions. The two
% functions (m.files)and this file should be under the same directory.
% Script written by Wenrong Cao, 2015-8-14. Comments improved in 2016-5.
% Contact information: caowenrong@gmail.com
clear
clf
Tau = [2 5 10 15]; % Define 4 different erosion response times (Tau)
% the code will test in each loop [unit: Myr]
TM = [0.24 0.087 0.53]; % Define 3 magmatic thickening ratio (beta) for
% Triassic [0.24], Jurassic [0.087]
% and Cretaceous [0.53] flare-ups
% beta = magma or pluton volume/initial volume
% of a crust
% Define 4 sets of deformation strain
% (thickening strain, epsilon_1 in text)
% negative = thinning, positive = thickening
STRAIN = [0 0 0 ; % Set 1: no strain
0.32 0.32 0.32 ; % Set 2: equal thickening strain
0.32 0 0.73 ; % Set 3: no Jurassic strain
0.32 -0.32 1.55 ]; % Set 4: Jurassic thinning/extension
TD = STRAIN (4,:); % choose one row in STRAIN matrix to test the
% tectonic thickening strain (TD)
% Set 4 (the fourth row) is selected by default
gamma = 2; % mass-of-root to mass-of-melt ratio. gamma=1: thin root,
% gamma=2 median root, gamma=3 thick root
% gamma=2 is selected by default
savename = 'S4G2'; % 'SAGB' format. A=which strain model you use.
% B=which gamma value you use
% This will be included in the file names of PDFs saved
% The acutal simulation does not need this parameter
% default savename = 'S4G2' (Set 4 strain+gamma=2)
for j= 1:4 % four loops, each loop tests a different Tau
H0 = 25; % initial thickness of crust, t=0, H0
rhoc = 2.8; % density of crust (10^3 kg/m3)
rhom = 3.3; % density of mantle (10^3 kg/m3)
rhor = 3.5; % denstiy of root (10^3 kg/m3)
C1 = (1-rhoc/rhom); % constant 1
C2 = (1-rhor/rhom); % constant 2
%-----------------------------------------------------------------------
% Set the parameters for magmatism, deformation, erosion, arc root for
% each magmatic flare-up and deformational event
%-----------------------------------------------------------------------
% Triassic event
Tau_1 = Tau(j); % erosion response time (Myr)
TM_1 = TM(1) ; % magmatic thickening for Triassic
TD_1 = TD(1); % deformational thickening for Triassic
TT_1 = total_thickness (TM_1, TD_1);
% calculate the total crustal thickenss based on TM and TD
gamma_1 = gamma; % mass-of-root to mass-of-melt ratio
t_r_f_1 = 190 ; % root foundering time for Triassic flare-up(Ma)
% Jurassic event
Tau_2 = Tau(j); % erosion response time (Myr)
TM_2 = TM(2) ; % magmatic thickening for Jurassic
TD_2 = TD(2); % deformational thickening for Jurassic
TT_2 = total_thickness (TM_2, TD_2);
% calculate the total crustal thickenss based on TM and TD
gamma_2 = gamma; % mass-of-root to mass-of-melt ratio
t_r_f_2 = 150 ; % root foundering time for Jurassic flare-up(Ma)
% Cretacesou event
Tau_3 = Tau(j); % erosion response time (Myr)
TM_3 = TM(3); % magmatic thickening for Cretaceous
TD_3 = TD(3); % deformational thickening for Cretaceous
TT_3 = total_thickness (TM_3, TD_3);
% calculate the total crustal thickenss based on TM and TD
gamma_3 = gamma; % mass-of-root to mass-of-melt ratio
t_r_f_3 = 5 ; % root foundering time for Cretaceous flare-up(Ma)
% -------------------------------------
% define time peroid for Triassic event
% -------------------------------------
t_m_s_1 = 250 ; % time when magmatic flare-up starts (Ma)
t_m_e_1 = 190 ; % time when magmatic flare-up ends (Ma)
t_d_s_1 = 250 ; % time when deformation starts (Ma)
t_d_e_1 = 190 ; % time when deformation ends (Ma)
t_r_s_1 = 250 ; % time when root formation starts (Ma)
t_r_e_1 = 190 ; % time when root formation ends (Ma)
% convert to forward time (will be convert back to Ma when plot)
t_m_s_1 = 250- t_m_s_1;
t_m_e_1 = 250- t_m_e_1;
t_d_s_1 = 250- t_d_s_1;
t_d_e_1 = 250- t_d_e_1;
t_r_s_1 = 250- t_r_s_1;
t_r_e_1 = 250- t_r_e_1;
t_r_f_1 = 250- t_r_f_1;
% -------------------------------------
% define time peroid for Jurassic event
% -------------------------------------
t_m_s_2 = 180 ; % time when magmatic flare-up starts (Ma)
t_m_e_2 = 150 ; % time when magmatic flare-up ends (Ma)
t_d_s_2 = 180 ; % time when deformation starts (Ma)
t_d_e_2 = 150 ; % time when deformation ends (Ma)
t_r_s_2 = 180 ; % time when root formation starts (Ma)
t_r_e_2 = 150 ; % time when root formation ends (Ma)
% convert to forward time (will be convert back to Ma when plot)
t_m_s_2 = 250- t_m_s_2;
t_m_e_2 = 250- t_m_e_2;
t_d_s_2 = 250- t_d_s_2;
t_d_e_2 = 250- t_d_e_2;
t_r_s_2 = 250- t_r_s_2;
t_r_e_2 = 250- t_r_e_2;
t_r_f_2 = 250- t_r_f_2;
% ---------------------------------------
% define time peroid for Cretaceous event
% ---------------------------------------
t_m_s_3 = 140 ; % time when magmatic flare-up starts (Ma)
t_m_e_3 = 90 ; % time when magmatic flare-up ends (Ma)
t_d_s_3 = 140 ; % time when deformation starts (Ma)
t_d_e_3 = 90 ; % time when deformation ends (Ma)
t_r_s_3 = 140 ; % time when root formation starts (Ma)
t_r_e_3 = 90 ; % time when root formation ends (Ma)
% convert to forward time (will be convert back to Ma when plot)
t_m_s_3 = 250- t_m_s_3;
t_m_e_3 = 250- t_m_e_3;
t_d_s_3 = 250- t_d_s_3;
t_d_e_3 = 250- t_d_e_3;
t_r_s_3 = 250- t_r_s_3;
t_r_e_3 = 250- t_r_e_3;
t_r_f_3 = 250- t_r_f_3;
%--------------------------------------------------------------
% Solve ODE (ordinary differential equation) for Triassuc event
%--------------------------------------------------------------
% Define time frame to solve ODE
% Time frame for Traissic event: 250-180 Ma
t1_1 = 250; % start time (Ma)
t2_1 = 180; % end time (Ma)
% convert to forward time
t1_1 = 250-t1_1; % (Myr)
t2_1 = 250-t2_1; % (Myr)
deltaH0_1 = 0; % initial thickend crutal thickness before Triassic events
% Since initial crustal thickness at 250 Ma is set to
% 25 km, the thickend part is 0 km
% call episode.m function to sovle erosion rates, elevation, crustal
% thickness (thickend part of crust), arc root thickness
% in the time frame defined above
% Explanations on inputs/outputs of the episode function can be found in
% the episode.m file.
[time_1,H_1,R_1, h_1,E_dot_1,M_dot_1, T_dot_1, R_dot_1] = ...
episode (t1_1,t2_1, H0, deltaH0_1,Tau_1,t_m_s_1, t_m_e_1, ...
t_d_s_1, t_d_e_1, t_r_s_1, t_r_e_1, TM_1, TT_1, gamma_1, t_r_f_1);
%-----------------------------------
% Solve ODE for Jurassic event
%-----------------------------------
% Define time frame to solve ODE
% Time frame for Jurassic event: 180-140 Ma
t1_2 = 180; % start time (Ma)
t2_2 = 140; % end time (Ma)
% convert to forward time
t1_2 = 250-t1_2; % (Myr)
t2_2 = 250-t2_2; % (Myr)
ind = find (time_1 ==t2_1); % find the last value of H (thickened part of
% crust)
deltaH0_2 = H_1(ind); % initial thickend part of crust (depends on the
% left-over after solve the first ODE)
% call episode.m function to sovle rates, elevation, crustal thickness,
% arc root thickness in the ODE time frame defined above
[time_2,H_2,R_2, h_2,E_dot_2,M_dot_2, T_dot_2, R_dot_2] = ...
episode (t1_2,t2_2, H0+deltaH0_2,deltaH0_2,Tau_2,t_m_s_2, t_m_e_2, ...
t_d_s_2, t_d_e_2, t_r_s_2, t_r_e_2, TM_2, TT_2, gamma_2, t_r_f_2);
%-----------------------------------
% Solve ODE for Cretaceous event
%-----------------------------------
% Define time frame to solve ODE
% Time frame for Cretaceous-Cenozoic event: 140-0 Ma
t1_3 = 140; % start time (Ma)
t2_3 = 0; % end time (Ma)
% convert to forward time
t1_3 = 250-t1_3; % (Myr)
t2_3 = 250-t2_3; % (Myr)
ind = find (time_2 ==t2_2); % find last value of H (thickened part of
% crust)
deltaH0_3 = H_2(ind); % initial thickend part of crust (depends on
% the left-over after solving the 2nd ODE)
% call episode.m function to sovle rates, elevation, crustal thickness,
% arc root thickness in the ODE time frame defined above
[time_3,H_3,R_3, h_3,E_dot_3,M_dot_3, T_dot_3, R_dot_3] =...
episode (t1_3,t2_3, H0+deltaH0_3,deltaH0_3,Tau_3,t_m_s_3, t_m_e_3, ...
t_d_s_3, t_d_e_3, t_r_s_3, t_r_e_3, TM_3, TT_3, gamma_3, t_r_f_3);
%----------------------------------------
% combine results calculated from 3 ODEs
%----------------------------------------
time = [time_1 ; time_2; time_3]; % time (Myr), forward time
H = [ H_1 ; H_2; H_3]; % Thickened part of crust through time(km)
R = [ R_1 ; R_2; R_3]; % arc root thickness through time (km)
h = [ h_1 ; h_2; h_3]; % elevation through time (km)
E_dot = [ E_dot_1 ; E_dot_2; E_dot_3];% Erosion rate through time(km/Myr)
M_dot = [ M_dot_1 ; M_dot_2; M_dot_3]; % relative magmatic thickeing
% rate (Myr^-1)through time
T_dot = [ T_dot_1 ; T_dot_2; T_dot_3]; % relative total thickeing
% rate (Myr^-1)through time
% Assgin Erosion rate = 0 if value of erosion rate <0
id= find(E_dot<0);
E_dot(id)=0;
time = 250-time; % convert forward time back to Ma
%---------------------
% plot elevation
%---------------------
figure(1)
plot (time,h,'LineWidth',2);
hold all
xlabel ('Time (Ma)','FontSize',20);
ylabel ('Elevation (km)','FontSize',20);
legend;
xlim([0, 250]);
set(gca,'XTick',[0:20:250]);
ylim([-2, 7]);
set(gca,'YTick',[-2:1:7]);
%ylim([-1, 3]); % scale for strain 1 (S1) model
%set(gca,'YTick',[-1:1:3]); % scale for strain 1 (S1) model
title('Elevation vs Time','FontSize',20);
grid on;
saveas(gcf,[savename '_Elevation'], 'pdf');
%------------------------
% plot crustal thickness
%------------------------
figure(2)
plot (time,H+H0,'LineWidth',3,'LineWidth',2);
% final crustal thickness = thickened part + H0
hold all
xlabel ('Time (Ma)','FontSize',20);
ylabel ('Crustal thickness (km)','FontSize',20);
legend;
xlim([0, 250]);
set(gca,'XTick',[0:20:250]);
ylim([10, 80]);
set(gca,'YTick',[10:10:80]);
%ylim([20, 50]); % scale for strain 1 (S1) model
%set(gca,'YTick',[20:5:50]); % scale for strain 1 (S1) model
title('Crustal thickness vs Time','FontSize',20);
grid on;
saveas(gcf,[savename '_CrustThickness'], 'pdf');
%---------------------------------------------------------------
% plot total thickness of crust and arc root (not shown in text)
% --------------------------------------------------------------
figure(3)
plot (time,H+H0+R,'LineWidth',2);
% total thickness= crustal thickness+root thickness
hold all
xlabel ('Time (Ma)','FontSize',20);
ylabel ('Total thickness (km)','FontSize',20);
legend;
xlim([0, 250]);
set(gca,'XTick',[0:20:250]);
ylim([0, 110]);
set(gca,'YTick',[0:10:110]);
%ylim([0, 100]); % scale for strain 1 (S1) model
%set(gca,'YTick',[0:20:100]); % scale for strain 1 (S1) model
title('Total thickness vs Time','FontSize',20);
grid on;
saveas(gcf,[savename '_TotalThickness'], 'pdf');
%------------------------
% plot rates
%------------------------
figure (4)
plot (time, H0*M_dot,'r','LineWidth',2);
% plot magmatic thickening rate [km/Myr]=
% relative magmatic thickening rate [1/Myr] *H0 [km]
hold on
plot (time, H0*T_dot,'b','LineWidth',2);
% plot total thickening rate [km/Myr]=
% relative total thickening rate [1/Myr] *H0 [km]
hold on
plot (time, E_dot,'g','LineWidth',2);
% plot erosion rate
hold all
xlabel ('Time (Ma)','FontSize',20);
ylabel ('Rate (km/Myr)','FontSize',20);
legend(' Magmatic thickening rate','Total thickening rate','Erosion rate');
set(gca,'XTick',[0:20:250]);
ylim([-1, 3]);
set(gca,'YTick',[-1:0.5:3]);
%ylim([-0.25, 1]); % scale for strain 1 (S1) model
%set(gca,'YTick',[-0.5:0.25:1]); % scale for strain 1 (S1) model
title('Thickening rates vs Time','FontSize',20);
grid on;
saveas(gcf,[savename '_Rates'], 'pdf');
%---------------------------------------------
% plot exhumation and thickened crust
% --------------------------------------------
figure(5)
timesteps = size(time);
timesteps = timesteps(1);
exhumation=-cumtrapz(time,E_dot); % intergration of erosion rate
plot (time,exhumation,'LineWidth',2);
hold all
xlabel ('Time (Ma)','FontSize',20);
ylabel ('Thickness (km)','FontSize',20);
xlim([0, 250]);
set(gca,'XTick',[0:20:250]);
ylim([0, 60]);
set(gca,'YTick',[0:10:60]);
%ylim([0, 25]); % scale for strain 1 (S1) model
%set(gca,'YTick',[0:5:25]); % scale for strain 1 (S1) model
title('Cumulative exhumation','FontSize',20);
grid on;
saveas(gcf,[savename '_CumuExhumation'], 'pdf');
end