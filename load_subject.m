function [out_data, emg_data] = load_subject(subject_to_load)

%%%%%%%%%%%%%%%%%%%%                     %%%%%%%%%%%%%%%%%%%%
% TO DO: Include Thorax, Elbow and Shoulder data from Vicon %
% TO DO: Adapt code so EMG works with condition selection   %
%%%%%%%%%%%%%%%%%%%%                     %%%%%%%%%%%%%%%%%%%%

% LOAD_SUBJECT      Function to load data corresponding to a single subject.
%                   Because each sorce has a different number of datapoints
%                   per trial, we have to load each trial and perform
%                   Dynamic Time Warping on both signals. After that, both
%                   signals are merged into a single variable. 
%                   UPDATED: Now also loads the EMG data.
%
% INPUT
% subject_to_load:  String with the subject's name.
%
% OUTPUT
% out_data:         Kinematic data from the corresponding subject.
% emg_data:         EMG data from the corresponding subject.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          18/11/21
% LAST MODIFIED:    14/06/22

% {'CeramicMug_CeramicMug';'CeramicMug_Glass';'CeramicMug_MetalMug';
%'CeramicPlate_CeramicPlate';'CeramicPlate_MetalPlate';'CeramicPlate_PlasticPlate';
%'Cube_Cube';'Cube_Cylinder';'Cube_Triangle';
%'Cylinder_Cube';'Cylinder_Cylinder';'Cylinder_Triangle';
%'Fork_Fork';'Fork_Knife';'Fork_Spoon';
%'Glass_CeramicMug';'Glass_Glass';'Glass_MetalMug';
%'Knife_Fork';'Knife_Knife';'Knife_Spoon';
%'MetalMug_CeramicMug';'MetalMug_Glass';'MetalMug_MetalMug';
%'MetalPlate_CeramicPlate';'MetalPlate_MetalPlate';'MetalPlate_PlasticPlate';
%'PingPongBall_PingPongBall';'PingPongBall_SquashBall';'PingPongBall_TennisBall';
%'PlasticPlate_CeramicPlate';'PlasticPlate_MetalPlate';'PlasticPlate_PlasticPlate';
%'Spoon_Fork';'Spoon_Knife';'Spoon_Spoon';
%'SquashBall_PingPongBall';'SquashBall_SquashBall';'SquashBall_TennisBall';
%'TennisBall_PingPongBall';'TennisBall_SquashBall';'TennisBall_TennisBall';
%'Triangle_Cube';'Triangle_Cylinder';'Triangle_Triangle'}
  
file = [pwd,'/Data/', 'Cut_Data_', subject_to_load, '.mat'];
aux_data = load(file);

out_data = [];
emg_data = [];

for i = 1:numel(aux_data.haptic_exploration_data_cut.tasks)

    % Condition to select trials
%     if contains(aux_data.haptic_exploration_data_cut.tasks(i).experiment_name, {'CeramicPlate_','MetalPlate_','PlasticPlate_'})
        
        glove_trial = aux_data.haptic_exploration_data_cut.tasks(i).data(5).data;
        vicon_trial = aux_data.haptic_exploration_data_cut.tasks(i).data(8).data;
        
        emg_trial = aux_data.haptic_exploration_data_cut.tasks(i).data(6).data; % Only includes the 64 HD EMG

        % CLEAN TRIALS
        fields_to_remove = {'ThumbAb', 'MiddleIndexAb', 'RingMiddleAb', 'PinkieRingAb'};
        glove_clean = table2array(removevars(glove_trial, fields_to_remove));
        fields_to_select = {'UNIX_time', 'Index_Proj_J1_Z', 'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z', 'Thumb_Proj_J1_Z'};
        vicon_clean = vicon_trial{:,fields_to_select};

        % DTW
        [~, new_glove_time, new_vicon_time] = dtw(glove_clean(:,1), vicon_clean(:,1));
        new_glove_trial = glove_clean(new_glove_time, :);
        new_vicon_trial = vicon_clean(new_vicon_time, :);
        
        % EMG downsample
        if ~isempty(emg_trial)
            downsampled_emg = interp1( 1:size(emg_trial, 1), table2array(emg_trial(:,2:end)), linspace(1, size(emg_trial, 1), size(new_vicon_trial, 1)), 'spline');
        else
            a=1; %Some trials in some subjects are empty for the EMG. This is a temporary NO solution.
        end
        
%         close all
%         plot(abs(table2array(emg_trial(:,2))))
%         figure
%         plot(abs(downsampled_emg(:,1)))
        

%             close all;
%             figure_name = [subject_to_load ' Trial: ' aux_data.haptic_exploration_data.subjects.tasks(i).experiment_name];
%             figure('Name', figure_name);
%             subplot(2,1,1);
%             plot(glove_clean(:,1), 'b');
%             hold on;
%             plot(vicon_clean(:,1), 'r');
%             legend('Glove', 'Vicon', 'Location', 'best');
%     
%             subplot(2,1,2);
%             close all;
%             plot(new_glove_trial(:,1), 'b');
%             hold on;
%             plot(new_vicon_trial(:,1), 'r');
%             legend('Glove', 'Vicon', 'Location', 'best');
    
%             figure('Name', figure_name);
%             plot(vicon_clean(:,1), vicon_clean(:,2), '.b');
%             hold on;
%             plot(new_glove_trial(:,1), new_vicon_trial(:,2), 'r');
%             legend('Old Vicon', 'New Vicon', 'Location', 'best');

        % MERGE DATA
        labels = aux_data.haptic_exploration_data_cut.tasks(i).data(3).data;
        task = repmat(aux_data.haptic_exploration_data_cut.tasks(i).experiment_name, size(new_glove_trial(:,2:end),1),1);
        time = new_vicon_trial(:,1);
        if size(labels,1) ~= size(new_glove_trial(:,2:end),1)
    %        disp(['Size doesnt match ' subject_to_load ' i = ' num2str(i) ' (' num2str(size(labels,1)) ',' num2str(size(new_glove_trial(:,2:end),1)) ')']); 
            labels(end+1) = labels(end);
        end
%         new_trial = [new_glove_trial(:,2:end) new_vicon_trial(:,2:end)];
%         assert(isequal(size(new_glove_trial,1), size(new_vicon_trial,1), size(labels,1), size(task,1), size(time,1)));

        new_trial = [new_glove_trial(:,2:end) new_vicon_trial(:,2:end) labels task time];
%         new_trial = [new_glove_trial(:,2:end) labels task time];
        out_data = [out_data; new_trial];
        emg_data = [emg_data; downsampled_emg];
    
%     end % END for selecting condition
    
end

end