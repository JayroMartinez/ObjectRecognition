%% load data
clc;
clear all;
close all;

raw_data = false;
data_path = {'./data/'};

%% Define variables
subjects = {'Subject_3'};
% subjects = {'Subject_4'}; % 
% subjects = {'Subject_5'}; % 
% subjects = {'Subject_6'}; % 
% subjects = {'Subject_7'}; % 
% subjects = {'Subject_8'}; % 
% subjects = {'Subject_9'}; % 
% subjects = {'Subject_10'}; % 
% subjects = {'Subject_11'}; %
% subjects = {'Subject_12'}; %
% subjects = {'Subject_13'}; % 
% subjects = {'Subject_14'}; % 

% subjects = {'Subject_3', 'Subject_4', 'Subject_5', 'Subject_6', 'Subject_7', ...
%     'Subject_8', 'Subject_9', 'Subject_10', 'Subject_11', 'Subject_12', 'Subject_14'};

task_names = {'CeramicMug vs. CeramicMug', 'CeramicMug vs. Glass', 'CeramicMug vs. MetalMug', ...
    'CeramicPlate vs. CeramicPlate', 'CeramicPlate vs. MetalPlate', 'CeramicPlate vs. PlasticPlate', ...
    'Cube vs. Cube', 'Cube vs. Cylinder', 'Cube vs. Triangle', 'Cylinder vs. Cube', 'Cylinder vs. Cylinder', ...
    'Cylinder vs. Triangle', 'Fork vs. Fork', 'Fork vs. Knife', 'Fork vs. Spoon', 'Glass vs. CeramicMug', ...
    'Glass vs. Glass', 'Glass vs. MetalMug', 'Knife vs. Fork', 'Knife vs. Knife', 'Knife vs. Spoon', ...
    'MetalMug vs. CeramicMug', 'MetalMug vs. Glass', 'MetalMug vs. MetalMug', 'MetalPlate vs. CeramicPlate', ...
    'MetalPlate vs. MetalPlate', 'MetalPlate vs. PlasticPlate', 'PingPongBall vs. PingPongBall', ...
    'PingPongBall vs. SquashBall', 'PingPongBall vs. TennisBall', 'PlasticPlate vs. CeramicPlate', ...
    'PlasticPlate vs. MetalPlate', 'PlasticPlate vs. PlasticPlate', 'Spoon vs. Fork', 'Spoon vs. Knife', ...
    'Spoon vs. Spoon', 'SquashBall vs. PingPongBall', 'SquashBall vs. SquashBall', 'SquashBall vs. TennisBall', ...
    'TennisBall vs. PingPongBall', 'TennisBall vs. SquashBall', 'TennisBall vs. TennisBall', ...
    'Triangle vs. Cube', 'Triangle vs. Cylinder', 'Triangle vs. Triangle'};

tasks = {'CeramicMug_CeramicMug', 'CeramicMug_Glass', 'CeramicMug_MetalMug', ...
    'CeramicPlate_CeramicPlate', 'CeramicPlate_MetalPlate', 'CeramicPlate_PlasticPlate', ...
    'Cube_Cube', 'Cube_Cylinder', 'Cube_Triangle', 'Cylinder_Cube', 'Cylinder_Cylinder', ...
    'Cylinder_Triangle', 'Fork_Fork', 'Fork_Knife', 'Fork_Spoon', 'Glass_CeramicMug', ...
    'Glass_Glass', 'Glass_MetalMug', 'Knife_Fork', 'Knife_Knife', 'Knife_Spoon', ...
    'MetalMug_CeramicMug', 'MetalMug_Glass', 'MetalMug_MetalMug', 'MetalPlate_CeramicPlate', ...
    'MetalPlate_MetalPlate', 'MetalPlate_PlasticPlate', 'PingPongBall_PingPongBall', ...
    'PingPongBall_SquashBall', 'PingPongBall_TennisBall', 'PlasticPlate_CeramicPlate', ...
    'PlasticPlate_MetalPlate', 'PlasticPlate_PlasticPlate', 'Spoon_Fork', 'Spoon_Knife', ...
    'Spoon_Spoon', 'SquashBall_PingPongBall', 'SquashBall_SquashBall', 'SquashBall_TennisBall', ...
    'TennisBall_PingPongBall', 'TennisBall_SquashBall', 'TennisBall_TennisBall', ...
    'Triangle_Cube', 'Triangle_Cylinder', 'Triangle_Triangle'};

list_joints_vicon = {'Index flex', 'Middle flex', 'Ring flex', 'Pinkie flex', 'Thumb flex'};
nbr_joints_vicon = { 18,           27,            24,          21,             32};

list_joints_glove = {'Index flex', 'Middle flex', 'Ring flex', 'Pinkie flex', 'Thumb flex'};
nbr_joints_glove = { 6,            8,             11,          14,             3};  

figure_counter = 1;

%% note:
%       CG_finger_MPJ = V_finger_Abs_J1_Z = ~ V_finger_Proj_J1_Y
%       CG_finger_finger_Ab ~ V_finger_J1_Z

super_init_per_object = true;
init_task = true;
init_graph = true;
init_graph_per_object = true;
init_count_eps = true;
object_counter = 1;
%% plot data
for p=1:length(subjects)
    disp(subjects{p})
    % load data per subject
    file_name_data_storage = strcat({'/home/simon/Code/human_haptic_exploration/data/'}, subjects{p}, {'/haptic_exploration/Cut_Data_'}, subjects{p}, {'.mat'});
    load(file_name_data_storage{:});
        
    for q=1:length(tasks)
        disp(tasks{q})
        display = ['Task ', num2str(q), ' of ', num2str(length(tasks(:)))];
        disp(display)
        
        %% write data to variable     
        CyberGlove = haptic_exploration_data_cut.tasks(q).data(5).data;
%         CyberGlove{:,05} = CyberGlove{:,05}-90; % Thumb Ab
        CyberGlove{:,02} = CyberGlove{:,02}+90; % Thumb Rot
        CyberGlove{:,03} = CyberGlove{:,03}-45; % Thumb MPJ
        CyberGlove{:,04} = CyberGlove{:,04}-90; % Thumb PIJ
        CyberGlove = deg2rad(CyberGlove{:,2:end}); % exclude time
%         Vicon = haptic_exploration_data_cut.tasks(q).data(12).data;
%         Vicon = deg2rad(Vicon{:,2:end});
        % Glove
        hand_val = [
            CyberGlove(:,04),...   % Thumb Ab
            CyberGlove(:,01),...   % Thumb Rot
            CyberGlove(:,02),...   % Thumb MPJ Flex
            CyberGlove(:,03),...   % Thumb  Ij (4)
            CyberGlove(:,09)/2,... % Index MPJ Ab
            CyberGlove(:,07),...   % Index MPJ Flex
            CyberGlove(:,06),...   % Index PIJ
            CyberGlove(:,06),...   % Index DIP
            CyberGlove(:,12)/2,... % Middle MPJ Ab
            CyberGlove(:,07),...   % Middle MPJ Flex
            CyberGlove(:,08),...   % Middle PIJ
            CyberGlove(:,08),...   % Middle DIP
            CyberGlove(:,15)/2,... % Ring MPJ Ab
            CyberGlove(:,10),...   % Ring MPJ Flex
            CyberGlove(:,11),...   % Ring PIJ
            CyberGlove(:,11),...   % Ring DIP
            CyberGlove(:,15)/2,... % Pinkie MPJ Ab
            CyberGlove(:,13),...   % Pinkie MPJ Flex
            CyberGlove(:,14),...   % Pinkie PIJ
            CyberGlove(:,14)];     % Pinkie DIP
        
        ha_raw = SGparadigmatic;
        %% PLOT
%         figure()
        for timepoints=1:length(hand_val(:,1))/5
            figure(1)
%             hold on
            hand_raw = SGmoveHand(ha_raw, hand_val(timepoints*5,:));
            SGplotHand(hand_raw);
%             view(-45,-45)
            v = [-45 0 -45];
            view(v)
            xlabel('x')
            ylabel('y')
            zlabel('z')
%             xlim([-10 150])
            ylim([20 140])
%             zlim([0 130])
            title(task_names{q})
        end
    end
end