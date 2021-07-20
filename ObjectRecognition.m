function ObjectRecognition

% OBJECTRECOGNITION This is the main function for the Object Recognition
% Project. 
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          17/06/21
% LAST MODIFIED:    20/07/21

% TODO:
%       - Variance explained plots
%       - Calculate (cluster) Synergies
%       - Plot Synergies

%% DATA LOADING

subjects_to_load = {'Subject_3';'Subject_4'};

all_data = {};

for i = 1:numel(subjects_to_load)
    
    all_data{i} = load_subject(subjects_to_load{i});
   
end

%% PCA CALCULATION FOR EACH SUBJECT

pca_values = {};

for j = 1:numel(all_data)
   
    [coeff, scores, explained] = pca_calculation(table2array(all_data{j}));
    
    pca_values{j,1} = coeff;
    pca_values{j,2} = scores;
    pca_values{j,3} = explained;
    
    clear coeff scores explained;
    
end

%% PCA CALCULATION FOR ALL SUBJECTS

all_subjects = [];

for k = 1:numel(pca_values(:,1))

    [all_subjects] = [all_subjects; table2array(all_data{k})];

end

[all_subjects_coeff, all_subjects_scores, all_subjects_explained] = pca_calculation(all_subjects);

%% VARIANCE PLOTS

% variance_plots(subjects_to_load, pca_values(:,3), all_subjects_explained);

%% CLUSTERING

% Here we change oder. PCS coeffs are organized as Joints x PCs (rows x
% columns) and we want PCs x Joints (note that PC1 from a subject comes the
% row after PC18 of previous subject.
pcs = [];

for l = 1:numel(subjects_to_load)
   
    pcs = [pcs; cell2mat(pca_values(l,1))'];
    
end

% Add coeffs calculated for all subjects together
pcs = [pcs; all_subjects_coeff'];

clustering(pcs, numel(subjects_to_load));

end