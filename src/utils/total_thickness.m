% Matlab script to solve the total crustal thickening due to
% simultaneous magma intrusion and deformation based on simple mass balance
% this function will be called in the main code MDE_model.m
% this function and MDE_model.m should be placed under the same directory
% Wenrong Cao, 2015-9-16. Comments improved in 2016-5
function [TT] = total_thickness(TM,TD)
% TM and TD are magmatic and deformation thickening strain, respectively
% TT is total thickening strain as output
% Alreadly take pluton strain and host rock volume loss into accout
% in the main code when the epsilon_1 is calculated so
% volume loss in host rock are set to zero here.
V_hostrocks_loss=0;
step = 200 ; % how many step you want to preform. Magma and deformation are
% added to the crust incrementally. Step number should be >
% 100 to have a good result resolution. When step number
% increases, the results converge.
crust_thickness_initial = 25; % initial arc crustal thickness (km)
crust_width_initial = 150; % initial arc crustal width (km)
% Initial crustal thickness and width
% doesn't affect the calculate.
% Put them here for
% understanding the code.
H0 = crust_thickness_initial; % initial thickness of crust.
L0 = crust_width_initial; % initial lenght of crust
V0 = H0*L0; % initial volumne of crust
% calculate host rock finite shortning strain based on TD
epsilon_finite = (1-V_hostrocks_loss)/(1+TD)-1;
% incremental strain (natural strain) of each step for host rocks
epsilon_step = (1+epsilon_finite)^(1/step)-1;
beta = TM; % finite volume fraction of magma
% incremental magma volume for each step
beta_step = V0*beta/step;
n=0; % counter for steps
m=0; % counter for magma increments
while n<=step % internal loop to solve thickness
% Step 1: Add magma increment first
V1= V0+beta_step; % volume increases
% V1: volume of crust after magma intrusion
L1 = L0 ; % arc width does not change
% L1: length of crust after magma intrusion
H1= V1/L1; % calculate arc crust thickness
% using mass balanace
% H1: thickness of crust after magma intrusion
m = m+1; % magmatic increment +1
% Step 2: Incrementally strain the crust
% calculate the arc width (L2) after deformaton
% only strain host rocks. Pluton remain unstrained
L2=(beta_step*m/V1)*L1... % length made by plutons (not strained)
+(1-beta_step*m/V1)*L1*(1+epsilon_step); % length made by host rock
% and subjected to strain
% calculate host rock volume after volume loss
V2_hostrocks = (V1-beta_step*m)... % host rock volume
*(1-V_hostrocks_loss); % volume loss precentage
% calculate pluton volume
V2_plutons = beta_step*m;
% calculate arc crust thickness using mass balanace
V2 = V2_plutons+V2_hostrocks;
H2 = V2/L2; % use mass balance to calculate new thickness
n=n+1;
% update parameters for next iteration
H0 = H2;
L0 = L2;
V0 = V2;
end
TT = (H2-crust_thickness_initial)/crust_thickness_initial;
% calculate the total thickning strain
end