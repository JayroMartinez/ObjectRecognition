function [out_data] = load_subject(subject_to_load)

%%%%%%%%%%%%%%%%%%%%                     %%%%%%%%%%%%%%%%%%%%
% TO DO: Include Thorax, Elbow and Shoulder data from Vicon %
%%%%%%%%%%%%%%%%%%%%                     %%%%%%%%%%%%%%%%%%%%

% LOAD_SUBJECT      Function to load data corresponding to a single subject.
%                   Because each sorce has a different number of datapoints
%                   per trial, we have to load each trial and perform
%                   Dynamic Time Warping on both signals. After that, both
%                   signals are merged into a single variable.
%
% INPUT
% subject_to_load:  String with the subject's name.
%
% OUTPUT
% out_data:         Data from the corresponding subject.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          18/11/21
% LAST MODIFIED:    05/01/22

  
file = [pwd,'/Data/', 'Cut_Data_', subject_to_load, '.mat'];
aux_data = load(file);

out_data = [];

for i = 1:numel(aux_data.haptic_exploration_data.subjects.tasks)

    glove_trial = aux_data.haptic_exploration_data.subjects.tasks(i).data(5).data;
    vicon_trial = aux_data.haptic_exploration_data.subjects.tasks(i).data(8).data;
    
    % CLEAN TRIALS
    fields_to_remove = {'ThumbAb', 'MiddleIndexAb', 'RingMiddleAb', 'PinkieRingAb'};
    glove_clean = table2array(removevars(glove_trial, fields_to_remove));
    fields_to_select = {'UNIX_time', 'Index_Proj_J1_Z', 'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z', 'Thumb_Proj_J1_Z'};
    vicon_clean = vicon_trial{:,fields_to_select};

    % DTW
    [~, new_glove_time, new_vicon_time] = dtw(glove_clean(:,1), vicon_clean(:,1));
    new_glove_trial = glove_clean(new_glove_time, :);
    new_vicon_trial = vicon_clean(new_vicon_time, :);

%         close all;
%         figure_name = [subject_to_load ' Trial: ' aux_data.haptic_exploration_data.subjects.tasks(i).experiment_name];
%         figure('Name', figure_name);
%         subplot(2,1,1);
%         plot(glove_clean(:,1), 'b');
%         hold on;
%         plot(vicon_clean(:,1), 'r');
%         legend('Glove', 'Vicon', 'Location', 'best');
% 
%         subplot(2,1,2);
%         plot(new_glove_trial(:,1), 'b');
%         hold on;
%         plot(new_vicon_trial(:,1), 'r');
%         legend('Glove', 'Vicon', 'Location', 'best');
% 
%         figure('Name', figure_name);
%         plot(vicon_clean(:,1), vicon_clean(:,2), '.b');
%         hold on;
%         plot(new_glove_trial(:,1), new_vicon_trial(:,2), 'r');
%         legend('Old Vicon', 'New Vicon', 'Location', 'best');

    % MERGE DATA
    new_trial = [new_glove_trial(:,2:end) new_vicon_trial(:,2:end)];
    out_data = [out_data; new_trial];
    
end

end