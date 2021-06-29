function ObjectRecognition

% OBJECTRECOGNITION This is the main function for the Object Recognition
% Project. 
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          17/06/21
% LAST MODIFIED:    24/06/21

%% DATA LOADING

subjects_to_load = {'Subject3';'Subject4'};

all_data = [];

for i = 1:numel(subjects_to_load)
    
    aux_data = load_subject(subjects_to_load{i});
    all_data = [all_data; aux_data];
    
    clear aux_data;
end

%% PCA CALCULATION

clean_data = all_data(:,9:end);

norm_data = normalize(clean_data);

[coeff,score,latent,tsquared,explained,mu] = pca(table2array(clean_data)); % Calculate the PCA for the data

[n_coeff,n_score,n_latent,n_tsquared,n_explained,n_mu] = pca(table2array(norm_data), 'Centered', false); % Calculate the PCA for the data


figure;
hold on;
plot(cumsum(explained), 'r', 'DisplayName', 'Not Normalized Cumulative Variance');
plot(cumsum(n_explained), 'b', 'DisplayName', 'Normalized Cumulative Variance');
b = bar([explained,n_explained]);
b(1).FaceColor = 'red';
b(2).FaceColor = 'blue';
set(b, {'DisplayName'}, {'Not Normalized Variance'; 'Normalized Variance'} );
yline(95, 'k', 'DisplayName', '95% Variance');
legend('Location', 'best');


end