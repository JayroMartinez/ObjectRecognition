function ObjectRecognition

% OBJECTRECOGNITION This is the main function for the Object Recognition
% Project. 
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          17/06/21
% LAST MODIFIED:    18/11/21

clear all;
close all;
clc;

%% DATA LOADING

subjects_to_load = {'Subject_3';'Subject_4';'Subject_5';'Subject_6';'Subject_7';'Subject_8';'Subject_9';'Subject_10';'Subject_11';'Subject_14'};
% subjects_to_load = {'Subject_14'};

% Sources initialization
glove = logical(false);
vicon = logical(true);

sources = [glove vicon];

all_data = {};

for i = 1:numel(subjects_to_load)
    
    all_data{i} = load_subject(subjects_to_load{i}, sources);
   
end

%% PCA CALCULATION FOR EACH SUBJECT (also means and standard deviations)

pca_values = {};
means = [];
stdevs = [];

for j = 1:numel(all_data)
    
    aux_mean = [];
    aux_stdev = [];
    
    subject_data = table2array(all_data{j});
   
    [coeff, scores, explained] = pca_calculation(subject_data);
    
    pca_values{j,1} = coeff;
    pca_values{j,2} = scores;
    pca_values{j,3} = explained;
    
    aux_mean = mean(subject_data);
    aux_stdev = std(subject_data);
    
    means = [means; aux_mean];
    stdevs = [stdevs; aux_stdev];
    
    clear coeff scores explained aux_mean aux_stdev;
    
end

% joint_names = all_data{1}.Properties.VariableNames;
% random_function(pca_values, means, stdevs,joint_names);

%% PCA CALCULATION FOR ALL SUBJECTS

all_subjects = [];

for k = 1:numel(pca_values(:,1))

    [all_subjects] = [all_subjects; table2array(all_data{k})];

end

[all_subjects_coeff, all_subjects_scores, all_subjects_explained] = pca_calculation(all_subjects);

%% VARIANCE PLOTS

% variance_plots(subjects_to_load, pca_values(:,3), all_subjects_explained);

%% CLUSTERING AND SYNERGY CALCULATION

% Here we change oder. PCS coeffs are organized as Joints x PCs (rows x
% columns) and we want PCs x Joints (note that PC1 from a subject comes the
% row after PC18 of previous subject.
pcs = [];

for l = 1:numel(subjects_to_load)
   
    pcs = [pcs; cell2mat(pca_values(l,1))'];
    
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
    means = [means; mean(all_subjects)];
    stdevs = [stdevs; std(all_subjects)];
    pca_values{3,1} = all_subjects_coeff;
    pca_values{3,2} = all_subjects_scores;
    pca_values{3,3} = all_subjects_explained;
    subjects_to_load = [subjects_to_load;{'All'}];
else
    number_of_subjects = numel(subjects_to_load);
    expl_var = cell2mat(pca_values(:,3)');
    coeffs = pca_values(:,1);
end

synergies = clustering(pcs, number_of_subjects);

% SORT SYNERGIES
[sorted_syn,sorted_var] = sort_synergies(synergies,expl_var);

% % Because clustering is not giving good results for two subjects + both
% % subjects together, we want to perform a recursive clustering.
% number_of_clusters = size(pcs,1) / (number_of_subjects);
% r_synergies = []; % Array to be filled with synergies
% r_synergies = recursive_clustering(pcs, number_of_subjects, r_synergies, number_of_clusters);
% 
% % SORT SYNERGIES
% [sorted_r_syn,sorted_r_var] = sort_synergies(r_synergies,expl_var);

% % CLUSTER EVALUATION
qual_trad = cluster_evaluation(sorted_syn, pcs);
mean_trad = mean(qual_trad);
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

joint_names = all_data{1}.Properties.VariableNames;

% plot_synergies(sorted_syn, joint_names, subjects_to_load, coeffs);
% plot_synergies(sorted_r_syn, joint_names, subjects_to_load, coeffs);

random_function(sorted_syn, pca_values, means, stdevs, joint_names, subjects_to_load);
end