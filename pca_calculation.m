function [coeff, score, explained] = pca_calculation(data)

% PCA_CALCULATION Function to load data corresponding to a single subject.
%
% INPUT
% data:             Matrix with the kinematic data.
%
% OUTPUT
% coeff:            PCA coefficients (eigenvectors).
% score:            PCA scores (eigenvalues).
% explained:        Explained variance for each component.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          19/07/21
% LAST MODIFIED:    19/07/21

norm_data = normalize(data);

[coeff,score,latent,tsquared,explained,mu] = pca(norm_data, 'Centered', false); % Calculate the PCA for the data


end
