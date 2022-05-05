function [norm_coeff, norm_score, norm_explained,notnorm_coeff,notnorm_score,notnorm_explained] = pca_calculation(data)

% PCA_CALCULATION Function to load data corresponding to a single subject.
%
% INPUT
% data:             Matrix with the kinematic data.
%
% OUTPUT
% coeff:            PCA coefficients (eigenvectors). Columns represent PCs
%                   and rows represent the old variables.
% score:            PCA scores (coordinates of each point in PC space).
%                   Columns represent PCs and rows represents observations.
% explained:        Explained variance for each component.
%
% THIS FINCTION HAS BEEN UPDATED TO RETURN TWO GROUPS OF VALUES: PCA WITH
% AND WIHOUT NORMALIZATION
%
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          19/07/21
% LAST MODIFIED:    04/05/21

% NORMALIZED PCA
norm_data = normalize(data);
[norm_coeff,norm_score,~, ~, norm_explained, ~] = pca(norm_data, 'Centered', false); % Calculate the normalized PCA for the data

% NOT NORMALIZED PCA
[notnorm_coeff,notnorm_score,~, ~, notnorm_explained, ~] = pca(data, 'Centered', true); % Calculate the PCA for the data


end
