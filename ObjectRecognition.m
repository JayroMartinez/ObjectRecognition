function ObjectRecognition

% OBJECTRECOGNITION This is the main function for the Object Recognition
% Project. 
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          17/06/21
% LAST MODIFIED:    19/07/21

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

    [all_subjects] = [all_subjects; cell2mat(pca_values(k,1))];

end

[all_subjects_coeff, all_subjects_scores, all_subjects_explained] = pca_calculation(all_subjects);


%% CLUSTERING

pcs = [s3_coeff'; s4_coeff'; n_coeff'];
% pcs = [s3_coeff'; s4_coeff'];

% clusters = clusterdata(pcs,'Criterion', 'distance', 'Linkage', 'complete', 'MaxClust', 18);
% aux_clust = reshape(clusters, [18, 2]);
% tree = linkage(pcs,'complete');
% dendrogram(tree)
% 
% idx_med = kmedoids(pcs,18, 'Distance', 'sqEuclidean');
% aux_kmed = reshape(idx_med, [18, 2]);

[idx_means,C,sumd,D] = kmeans(pcs,18, 'Distance', 'sqEuclidean','EmptyAction','error','OnlinePhase','on','Replicates',5);
aux_kmean = reshape(idx_means, [18, 3]);
% aux_kmean = reshape(idx_means, [18, 2]);

[silh3,h] = silhouette(pcs,idx_means,'sqEuclidean');
xlabel('Silhouette Value')
ylabel('Cluster')
disp(['Media Clusters: ' string(mean(silh3))]);

% % k-nn as clustering (with k=1)
% response = [];
% for i = 1:18
%     aux_resp = ['PC' num2str(i)];
%     response = [response; string(aux_resp)];
% end
% model_n = fitcknn(n_coeff', response', 'NumNeighbors', 1);
% labels_s3 = predict(model_n, s3_coeff');
% labels_s4 = predict(model_n, s4_coeff');
% results_all_as_model = [response, string(labels_s3), string(labels_s4)];
% 
% model_s3 = fitcknn(s3_coeff', response', 'NumNeighbors', 1);
% labels_n = predict(model_s3, n_coeff');
% labels_s4 = predict(model_s3, s4_coeff');
% results_s3_as_model = [response, string(labels_n), string(labels_s4)];
% 
% model_s4 = fitcknn(s4_coeff', response', 'NumNeighbors', 1);
% labels_s3 = predict(model_s4, s3_coeff');
% labels_n = predict(model_s4, n_coeff');
% results_s4_as_model = [response, string(labels_s3), string(labels_n)];

a=1;


%% BARPLOTS SYNERGIES
joint_names = clean_data.Properties.VariableNames;
joint_names = regexprep(joint_names, '\w_', '');

% S1
figure;
b = bar([s3_coeff(:,1),s4_coeff(:,1)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 1');
legend('Location', 'best');


% S2
figure;
subplot(2,2,1);
b = bar([s3_coeff(:,2),s4_coeff(:,3)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 2');
% legend('Location', 'best');

% S3
% figure;
subplot(2,2,2);
b = bar([s3_coeff(:,3),s4_coeff(:,2)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 3');
% legend('Location', 'best');

% S4
% figure;
subplot(2,2,3);
b = bar([s3_coeff(:,4),s4_coeff(:,7)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 4');
% legend('Location', 'best');

% S5
% figure;
subplot(2,2,4);
b = bar([s3_coeff(:,5),s4_coeff(:,5)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 5');
legend('Location', 'best');

% S6
figure;
subplot(2,2,1);
b = bar([s3_coeff(:,6),s4_coeff(:,6)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 6');
% legend('Location', 'best');

% S7
% figure;
subplot(2,2,2);
b = bar([s3_coeff(:,7),s4_coeff(:,10)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 7');
% legend('Location', 'best');

% S8
% figure;
subplot(2,2,3);
b = bar([s3_coeff(:,8),s4_coeff(:,13)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 8');
% legend('Location', 'best');

% S9
% figure;
subplot(2,2,4);
b = bar([s3_coeff(:,9),s4_coeff(:,12)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 9');
legend('Location', 'best');

% S10
figure;
subplot(2,2,1);
b = bar([s3_coeff(:,10),s4_coeff(:,9)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 10');
% legend('Location', 'best');

% S11
% figure;
subplot(2,2,2);
b = bar([s3_coeff(:,11),s4_coeff(:,8)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 11');
% legend('Location', 'best');

% S12
% figure;
subplot(2,2,3);
b = bar([s3_coeff(:,12),s4_coeff(:,14)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 12');
% legend('Location', 'best');

% S13
% figure;
subplot(2,2,4);
b = bar([s3_coeff(:,13),s4_coeff(:,15)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 13');
legend('Location', 'best');

% S14
figure;
subplot(2,2,1);
b = bar([s3_coeff(:,14),s4_coeff(:,4)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 14');
% legend('Location', 'best');

% S15
% figure;
subplot(2,2,2);
b = bar([s3_coeff(:,15),s4_coeff(:,16)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 15');
% legend('Location', 'best');

% S16
% figure;
subplot(2,2,3);
b = bar([s3_coeff(:,16),s4_coeff(:,11)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 16');
% legend('Location', 'best');

% S17
% figure;
subplot(2,2,4);
b = bar([s3_coeff(:,17),s4_coeff(:,18)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 17');
legend('Location', 'best');

% S18
figure;
b = bar([s3_coeff(:,18),s4_coeff(:,17)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Synergy 18');
legend('Location', 'best');

end