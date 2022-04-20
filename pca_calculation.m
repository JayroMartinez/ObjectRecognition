function [coeff, score, explained] = pca_calculation(data)

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
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          19/07/21
% LAST MODIFIED:    03/11/21

norm_data = normalize(data);
[coeff,score,~, ~, explained, ~] = pca(norm_data, 'Centered', false); % Calculate the PCA for the data

% JUST FOR TESTING PURPOSES
% [coeff,score,~, ~, explained, mu] = pca(data, 'Centered', true); % Calculate the PCA for the data


end
