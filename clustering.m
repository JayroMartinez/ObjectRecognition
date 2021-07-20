function clustering(pcs, number_of_subjects)

% CLUSTERING Function to load data corresponding to a single subject.
%
% INPUT
% pcs:                  Matrix PCA coefficients. Each row represents a PC 
%                       from and each column represents a joint. Notice 
%                       that the number of rows is [number of subjects X 
%                       PCs per subject]. After last PC from a subject it 
%                       comes the first PC of the next subject. The last 
%                       group of PCs correspond to those calculated from 
%                       the data for all subjects together.
%
% number_of_subjects:   Number of subjects that has been loaded.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          19/07/21
% LAST MODIFIED:    20/07/21

number_of_synergies = size(pcs,1) / (number_of_subjects + 1);

[idx_means,C,sumd,D] = kmeans(pcs,number_of_synergies, 'Distance', 'sqEuclidean','EmptyAction','error','OnlinePhase','on','Replicates',25);
aux_kmean = reshape(idx_means, [number_of_synergies, number_of_subjects + 1]);

[silh3,h] = silhouette(pcs,idx_means,'sqEuclidean');
xlabel('Silhouette Value')
ylabel('Cluster')
disp(['Media Clusters: ' string(mean(silh3))]);

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