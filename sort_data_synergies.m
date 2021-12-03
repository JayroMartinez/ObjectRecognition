function sorted_pcs = sort_data_synergies(sorted_syn, pca_values)

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
% sorted_pca:   M x N x P couble array [Joints x Subjects x Synergies]. 
% sorted_means: 
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          03/12/21
% LAST MODIFIED:    03/12/21

sorted_pcs = [];

for syn = 1:size(sorted_syn,1)
    
    aux_pcs = [];
    for suj = 1:size(sorted_syn,2)
    
        if ~isnan(sorted_syn(syn, suj))
            pc_to_get = sorted_syn(syn, suj);
            new_pc = pca_values{suj, 1}(:,pc_to_get);
        else
            new_pc = NaN(size(pca_values{1},1),1);
        end

        aux_pcs = [aux_pcs new_pc];
        
    end
    
    sorted_pcs = cat(3,sorted_pcs, aux_pcs);
end

end