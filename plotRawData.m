function plotRawData

clear all;
close all;
clc;

%% DATA LOADING
subjects_to_load = {'Subject_3';'Subject_4';'Subject_5';'Subject_6';'Subject_7';'Subject_8';'Subject_9';'Subject_10';'Subject_11';'Subject_12';'Subject_14'};

data = {};
for i=1:numel(subjects_to_load)
    
    file = strcat(pwd,'/Data/', 'Cut_Data_', subjects_to_load(i), '.mat');
    aux_data = load(file{:});
    
    data{end+1} = aux_data;
    
end

%% GENERAL PAREMETERS

subject = 1;
task = 1;

number_plots = 5;

subject_name = strrep(subjects_to_load{subject}, '_', ' ');
task_sel = data{subject}.haptic_exploration_data_cut.tasks(task).experiment_name;
task_name = strrep(task_sel, '_', ' ');  
info = string(['ORIGINAL DATA' newline subject_name ' Task:' task_name]);

%% test trial
% test_vicon = data{1}.haptic_exploration_data_cut.tasks(20).data(8).data;
test_glove = data{subject}.haptic_exploration_data_cut.tasks(task).data(5).data;

test_glove{:,1} = 0;
glove_joint_list = [5, 2, 3, 4, 1, 6, 7, 7, 1, 8, 9, 9, 1, 11, 12, 12, 1, 14, 15, 15];
new_glove = test_glove{:,glove_joint_list};
new_glove(:,1) = new_glove(:,1)-45;
new_glove(:,2) = -new_glove(:,2)+45;
new_glove(:,3) = -new_glove(:,3);
new_glove(:,4) = -new_glove(:,4);


%% PLOT

ha = SGparadigmatic; % Define and create hand

time_to_plot = ceil(linspace(1,size(new_glove, 1), number_plots));

close all;
for iter = 1:numel(time_to_plot)
    
    figure;
    title(info);
    movement = deg2rad(new_glove(time_to_plot(iter), :));
    hand = SGmoveHand(ha, movement');
    
    SGplotHand(hand);

end

%% RECONSTRUCTION (JUST TO CHECK)

% subject = 1;
% task = 1;
% number_plots = 5;
% 
% % ORIGINAL DATA
% subject_data = all_data{subject};
% tasks = unique(subject_data(:,21));
% task_sel = tasks(task);
% task_idx = contains(subject_data(:,21),task_sel);
% raw_data = str2double(subject_data(task_idx, 1:19));
% empty = zeros(size(raw_data,1),1);
% % reordered_raw = [empty,raw_data(:,1),raw_data(:,2),raw_data(:,3),empty,raw_data(:,4),raw_data(:,5),raw_data(:,5),empty,raw_data(:,6),raw_data(:,7),raw_data(:,7),empty,raw_data(:,8),raw_data(:,9),raw_data(:,9),empty,raw_data(:,10),raw_data(:,11),raw_data(:,11)];
% reordered_raw = [empty,-raw_data(:,1)+45,-raw_data(:,2),-raw_data(:,3),empty,raw_data(:,4),raw_data(:,5),raw_data(:,5),empty,raw_data(:,6),raw_data(:,7),raw_data(:,7),empty,raw_data(:,8),raw_data(:,9),raw_data(:,9),empty,raw_data(:,10),raw_data(:,11),raw_data(:,11)];
% 
% ha = SGparadigmatic; % Define and create hand
% time_to_plot = ceil(linspace(1,size(raw_data, 1), number_plots));
% subject_name = strrep(subjects_to_load{subject}, '_', ' ');
% task_name = strrep(task_sel, '_', ' ');  
% info = strjoin(['ORIGINAL DATA' newline subject_name ' Task:' task_name]);
% 
% close all;
% 
% % for iter = 1:number_plots
% %     
% %     figure;
% %     title(info);
% %     movement = deg2rad(reordered_raw(time_to_plot(iter), :));
% %     hand = SGmoveHand(ha, movement');
% %     
% %     SGplotHand(hand);
% %     sgtitle(info);
% % 
% % end
% 
% % PCA DATA
% selected_pca_scores = pca_values{subject,2}(task_idx,:);
% reconstruction = selected_pca_scores * pca_values{subject,1}' + pca_values{subject,3}';
% 
% figure;
% plot(raw_data(1,:))
% hold on;
% plot(reconstruction(1,:))
% 
% a=1;


end