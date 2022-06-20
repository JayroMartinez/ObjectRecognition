function ObjectRecognition

% OBJECTRECOGNITION This is the main function for the Object Recognition
%                   Project. 
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          17/06/21
% LAST MODIFIED:    14/06/22

clear all;
close all;
clc;

%% DATA LOADING

% subjects_to_load = {'Subject_3';'Subject_4';'Subject_5';'Subject_6';'Subject_7';'Subject_8';'Subject_9';'Subject_10';'Subject_11';'Subject_12'};
% subjects_to_load = {'Subject_3';'Subject_4';'Subject_5';'Subject_6';'Subject_7';'Subject_8';'Subject_9';'Subject_10';'Subject_11';'Subject_12';'Subject_14'};
subjects_to_load = {'Subject_3';'Subject_4';'Subject_5';'Subject_6';'Subject_7';'Subject_8';'Subject_9';'Subject_10';'Subject_14'};
% subjects_to_load = {'Subject_3';'Subject_4';'Subject_5'};
% subjects_to_load = {'Subject_5'};

all_data = {};
emg_data = {};

for i = 1:numel(subjects_to_load)
    
    [all_data{i}, emg_data{i}] = load_subject(subjects_to_load{i});
   
end

clear i;

%% AUXILIAR CODE FOR EP SELECTION

[kinem_fil, ep_labels, task_labels, time, emg_fil] = filter_ep(all_data, emg_data, '');

%% SAVE DATA AS JSON (to be read by python)

path = [pwd '/PyCode/PyData'];

dat_kin_filtered = jsonencode(kinem_fil);
kin_path = fullfile(path, 'filtered_data.json');
kin_file = fopen(kin_path, 'w');
fprintf(kin_file, dat_kin_filtered);
clear data_kin_filtered kin_path kin_file;

dat_emg_filtered = jsonencode(emg_fil);
emg_path = fullfile(path, 'emg_data.json');
emg_file = fopen(emg_path, 'w');
fprintf(emg_file, dat_emg_filtered);
clear data_emg_filtered emg_path emg_file;

dat_ep = jsonencode(ep_labels);
ep_path = fullfile(path, 'ep_labels.json');
ep_file = fopen(ep_path, 'w');
fprintf(ep_file, dat_ep);
clear dat_ep ep_path ep_file;

dat_task = jsonencode(task_labels);
task_path = fullfile(path, 'task_labels.json');
task_file = fopen(task_path, 'w');
fprintf(task_file, dat_task);
clear dat_task task_path task_file;

%% PCA CALCULATION FOR EACH SUBJECT (also means and standard deviations)

pca_values = {};
% pca_notnorm = {};
means = [];
% stdevs = [];

for j = 1:numel(kinem_fil)
    
%     aux_mean = [];
%     aux_stdev = [];
    
    subject_data = kinem_fil{j};
   
%     [coeff, scores, explained] = pca_calculation(subject_data);
%     [norm_coeff, norm_score, norm_explained,notnorm_coeff,notnorm_score,notnorm_explained] = pca_calculation(subject_data);
    [norm_coeff, norm_score, norm_explained,~,~,~] = pca_calculation(subject_data);
    
%     disp('Number of NaNs BEFORE PCA for ' + string(subjects_to_load(j)) + ': ' + num2str(unique(sum(isnan(subject_data)))));
%     disp('Number of NaNs AFTER PCA for '+ string(subjects_to_load(j)) + ': ' + num2str(unique(sum(isnan(norm_score)))) + newline);
    
    pca_values{j,1} = norm_coeff;
    pca_values{j,2} = norm_score;
    pca_values{j,3} = norm_explained;
    
%     pca_notnorm{j,1} = notnorm_coeff;
%     pca_notnorm{j,2} = notnorm_score;
%     pca_notnorm{j,3} = notnorm_explained;
    
    aux_mean = mean(subject_data, 'omitnan');
%     aux_stdev = std(subject_data);
%     
    means = [means; aux_mean];
%     stdevs = [stdevs; aux_stdev];
    
%     clear coeff scores explained aux_mean;
    clear norm_coeff norm_score norm_explained aux_mean;
    clear notnorm_coeff notnorm_score notnorm_explained;
%     clear aux_mean aux_stdev;
    
end

clear j;


%% PCA CALCULATION FOR ALL SUBJECTS

% all_subjects = [];
% 
% for k = 1:numel(subjects_to_load)
% 
%     [all_subjects] = [all_subjects; str2double(all_data{k}(:,1:end-3))];
% 
% end
% 
% [all_subjects_coeff, all_subjects_scores, all_subjects_explained, ~, ~, ~] = pca_calculation(all_subjects);

%% VARIANCE PLOTS

% variance_plots(subjects_to_load, pca_values(:,3), []);

%% CLUSTERING AND SYNERGY CALCULATION

% Here we change oder. PCS coeffs are organized as Joints x PCs (rows x
% columns) and we want PCs x Joints (note that PC1 from a subject comes the
% row after last PC of previous subject.
pcs = [];
% notnorm_pcs = [];

for l = 1:numel(subjects_to_load)
   
    pcs = [pcs; cell2mat(pca_values(l,1))'];
%     notnorm_pcs = [notnorm_pcs; cell2mat(pca_notnorm(l,1))'];
    
end

% Logical that indicates if we include data from all subjects together
include_all_subjects = logical(false);
% If the data from all subjects together is included, we update the number
% of subjects and add coeffs calculated for all subjects together
if include_all_subjects
    number_of_subjects = numel(subjects_to_load) + 1;
    pcs = [pcs; all_subjects_coeff'];
    expl_var = [cell2mat(pca_values(:,3)'), all_subjects_explained];
    coeffs = [pca_values(:,1);{all_subjects_coeff}];
    subjects_to_load = [subjects_to_load;{'All Subjects'}];
    means = [means; mean(all_subjects, 'omitnan')];
%     stdevs = [stdevs; std(all_subjects)];
    pca_values{end+1,1} = all_subjects_coeff;
    pca_values{end,2} = all_subjects_scores;
    pca_values{end,3} = all_subjects_explained;
    subjects_to_load = [subjects_to_load;{'All'}];
else
    number_of_subjects = numel(subjects_to_load);
    
    expl_var = cell2mat(pca_values(:,3)');
    coeffs = pca_values(:,1);
    
%     notnorm_expl_var = cell2mat(pca_notnorm(:,3)');
%     notnorm_coeffs = pca_notnorm(:,1);
end

synergies = clustering(pcs, number_of_subjects);
% notnorm_synergies = clustering(notnorm_pcs, number_of_subjects);

%% SORT SYNERGIES
[sorted_syn,sorted_var] = sort_synergies(synergies,expl_var);
% [nn_sorted_syn,nn_sorted_var] = sort_synergies(notnorm_synergies,notnorm_expl_var);

% % Because clustering is not giving good results for two subjects + both
% % subjects together, we want to perform a recursive clustering.
% number_of_clusters = size(pcs,1) / (number_of_subjects);
% r_synergies = []; % Array to be filled with synergies
% r_synergies = recursive_clustering(pcs, number_of_subjects, r_synergies, number_of_clusters);
% 
% % SORT SYNERGIES
% [sorted_r_syn,sorted_r_var] = sort_synergies(r_synergies,expl_var);
% 
% % CLUSTER EVALUATION
% 
% qual_trad = cluster_evaluation(sorted_syn, pcs);
% mean_trad = mean(qual_trad);
% 
% qual_rec = cluster_evaluation(sorted_r_syn, pcs);
% mean_rec = mean(qual_rec);
% 
% max_y = max([qual_trad, qual_rec],[],'all');
% figure;
% b = bar([qual_trad, qual_rec]);
% set(b, {'DisplayName'}, [{'Modified Traditional'}; {'Recursive'}]);
% text(1, max_y - 0.05, ['Modified Traditional Mean: ' num2str(round(mean_trad,3))]);
% text(1, max_y - 0.08, ['Recursive Mean: ' num2str(round(mean_rec,3))]);
% legend('Location', 'best', 'Interpreter', 'none');


%% SYNERGY PLOTS

% barplot_synergies(sorted_syn, joint_names, subjects_to_load, coeffs);
% barplot_synergies(sorted_r_syn, joint_names, subjects_to_load, coeffs);

[sorted_pcs, sorted_scores, sorted_variances] = sort_data_synergies(sorted_syn, pca_values);
% [nn_sorted_pcs, nn_sorted_scores, nn_sorted_variances] = sort_data_synergies(nn_sorted_syn, pca_notnorm);

% synergy_to_plot = 1;
% handplot_synergies(sorted_pcs, means, synergy_to_plot, subjects_to_load);


%% SYNERGY VARIANCE CALCULATION

% synergy_variances = syn_variance_calculation(sorted_pcs);
% synergy_variances = syn_variance_calculation_oneTOone(sorted_pcs);

%% SYNERGY EVOLUTION
evolution_comparison(sorted_scores, sorted_pcs, means, ep_labels, task_labels, time, subjects_to_load, all_data);
% evolution_comparison(nn_sorted_scores, nn_sorted_pcs, means, ep_labels, task_labels, time, subjects_to_load, all_data);

%% CLASSIFICATION

classification_function(kinem_fil, ep_labels, task_labels);

end