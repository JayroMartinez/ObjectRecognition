function ObjectRecognition

% OBJECTRECOGNITION This is the main function for the Object Recognition
% Project. 
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          17/06/21
% LAST MODIFIED:    01/07/21

%% DATA LOADING

subjects_to_load = {'Subject3';'Subject4'};

all_data = [];

for i = 1:numel(subjects_to_load)
    
    aux_data = load_subject(subjects_to_load{i});
    all_data = [all_data; aux_data];
    
    clear aux_data;
end

% Subjects data apart
s3data = load_subject('Subject3');
s4data = load_subject('Subject4');

%% PCA CALCULATION

clean_data = all_data(:,9:end);

norm_data = normalize(clean_data);

% [coeff,score,latent,tsquared,explained,mu] = pca(table2array(clean_data)); % Calculate the PCA for the data

[n_coeff,n_score,n_latent,n_tsquared,n_explained,n_mu] = pca(table2array(norm_data), 'Centered', false); % Calculate the PCA for the data
% 
% 
% figure;
% hold on;
% plot(cumsum(explained), 'r', 'DisplayName', 'Not Normalized Cumulative Variance');
% plot(cumsum(n_explained), 'b', 'DisplayName', 'Normalized Cumulative Variance');
% b = bar([explained,n_explained]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Not Normalized Variance'; 'Normalized Variance'} );
% yline(85, 'c', 'DisplayName', '85% Variance');
% yline(90, 'm', 'DisplayName', '90% Variance');
% yline(95, 'k', 'DisplayName', '95% Variance');
% legend('Location', 'best');

% Splitted calculations
s3_clean_data = s3data(:,9:end);
s4_clean_data = s4data(:,9:end);

s3_norm_data = normalize(s3_clean_data);
s4_norm_data = normalize(s4_clean_data);

[s3_coeff,s3_score,s3_latent,s3_tsquared,s3_explained,s3_mu] = pca(table2array(s3_norm_data)); % Calculate the PCA for the data
[s4_coeff,s4_score,s4_latent,s4_tsquared,s4_explained,s4_mu] = pca(table2array(s4_norm_data)); % Calculate the PCA for the data

figure;
hold on;
plot(cumsum(s3_explained), 'r', 'DisplayName', 'Subject 3 Cumulative Variance');
plot(cumsum(s4_explained), 'b', 'DisplayName', 'Subject 4 Cumulative Variance');
plot(cumsum(n_explained), 'g', 'DisplayName', 'All Subjects Cumulative Variance');
b = bar([s3_explained,s4_explained,n_explained]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
b(3).FaceColor = 'green';
set(b, {'DisplayName'}, {'Subject 3 Variance'; 'Subject 4 Variance'; 'All Subjects Variance'} );
yline(85, 'c', 'DisplayName', '85% Variance');
yline(90, 'm', 'DisplayName', '90% Variance');
yline(95, 'k', 'DisplayName', '95% Variance');
ylim([0 100]);
legend('Location', 'best');


%% DATA PROJECTED INTO PCS 1-3 LAYING IN THE ORIGINAL SPACE
% reconstruction_pc1 = n_score(:,1) * n_coeff(:,1)';
% reconstruction_pc2 = n_score(:,2) * n_coeff(:,2)';
% reconstruction_pc3 = n_score(:,3) * n_coeff(:,3)';
% 
% figure;
% hold on;
% scatter3(table2array(norm_data(:,1)),table2array(norm_data(:,2)),table2array(norm_data(:,3)), 'y.');
% plot3(reconstruction_pc1(:,1),reconstruction_pc1(:,2),reconstruction_pc1(:,3), 'b');
% plot3(reconstruction_pc2(:,1),reconstruction_pc2(:,2),reconstruction_pc2(:,3), 'g');
% plot3(reconstruction_pc3(:,1),reconstruction_pc3(:,2),reconstruction_pc3(:,3), 'r');
% daspect([1 1 1]);
% grid on;

%% BARPLOTS SYNERGIES
joint_names = clean_data.Properties.VariableNames;
joint_names = regexprep(joint_names, '\w_', '');

% S1
figure;
b = bar([s3_coeff(:,1),s4_coeff(:,1),n_coeff(:,1)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
b(3).FaceColor = 'green';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'; 'All Subjects'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('First Synergy');
legend('Location', 'best');


figure;
% S5
subplot(2,2,4);
b = bar([s3_coeff(:,5),s4_coeff(:,5),n_coeff(:,5)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
b(3).FaceColor = 'green';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'; 'All Subjects'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Fifth Synergy');

% S2
subplot(2,2,1);
b = bar([s3_coeff(:,2),s4_coeff(:,2),n_coeff(:,2)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
b(3).FaceColor = 'green';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'; 'All Subjects'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Second Synergy');

% S3
subplot(2,2,2);
b = bar([s3_coeff(:,3),s4_coeff(:,3),n_coeff(:,3)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
b(3).FaceColor = 'green';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'; 'All Subjects'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Third Synergy');

% S4
subplot(2,2,3);
b = bar([s3_coeff(:,4),s4_coeff(:,4),n_coeff(:,4)]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
b(3).FaceColor = 'green';
set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'; 'All Subjects'});
set(gca,'xtick',1:numel(joint_names));
set(gca,'XTickLabel',joint_names);
xtickangle(45);
title('Fourth Synergy');

legend('Location', 'best');




end