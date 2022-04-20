function [sorted_pcs, sorted_scores, sorted_variances]  = sort_data_synergies(sorted_syn, pca_values)

% SORT_DATA_SYNERGIES Reorganize the data based on synergies order.
%
% INPUT
% sorted_syn:  M x N double array [Subjects x Synergies]. Each row 
%              represents a synergy, each column represents a subject.
%              Value in (i,j) means that, for subject 'j', synergy 'i' 
%              corresponds to PC(p(i,j)). 
% pca_values:  M x 3 cell array. Each row corresponds to a subject. Columns
%              correspond to PCA coefficients, scores and explained
%              variance.
%
% OUTPUT
% sorted_pcs:       M x N x P double array [Joints x Subjects x Synergies].
%                   Value (m,n,p) correspond to the joint 'm' for subject 'n'
%                   and synergy 'p'. Could contain NaN values.
%
% sorted_scores:    P x N cell array [Synergies x Subjects]. Each position
%                   in the cell array (p,n) represents the scores (one 
%                   value per datapoint)for synergy 'p' and subject 'n'.
%                   Could contain NaN values.
%
% sorted_variances: P x N double array [Synergies x Subjects]. Each position
%                   in the array (p,n) represents the scores variance
%                   explained by synergy 'p' in subject 'n'. 
%                   Could contain NaN values.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          03/12/21
% LAST MODIFIED:    30/03/22

sorted_pcs = [];
sorted_scores = {};
sorted_variances = [];

for syn = 1:size(sorted_syn,1)
    
    aux_pcs = [];
    aux_scores = {};
    aux_variances = [];
    
    for suj = 1:size(sorted_syn,2)
    
        if ~isnan(sorted_syn(syn, suj))
            pc_to_get = sorted_syn(syn, suj);
            new_pc = pca_values{suj, 1}(:,pc_to_get);
            new_scores = pca_values{suj, 2}(:,pc_to_get);
            new_variance = pca_values{suj, 3}(pc_to_get);
        else
            new_pc = NaN(size(pca_values{suj,1},1),1);
            new_scores = NaN(size(pca_values{suj,2},1),1);
            new_variance = NaN;
        end

        aux_pcs = [aux_pcs new_pc];
        aux_scores{end+1} = new_scores;
        aux_variances = [aux_variances new_variance];
        a=1;
    end
    
    sorted_pcs = cat(3,sorted_pcs, aux_pcs);
    sorted_scores = cat(1,sorted_scores, aux_scores);
    sorted_variances = cat(1, sorted_variances, aux_variances);
end

end