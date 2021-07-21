function [synergies] = recursive_clustering(pcs, number_of_subjects, synergies)

%%%%%%% WARNING: after creating the synergies we have to sort it depending
%%%%%%% on the eigenvalues

% RECURSIVE_CLUSTERING Function to load data corresponding to a single subject.
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
% synergies:            Array to be filled with the synergies. Each value
%                       in position (i,j) represents the PC for subject 'j'
%                       that belongs to synergy 'i'.
%
% OUTPUT
% synergies:            Updated array to be filled with the synergies. Each
%                       value in position (i,j) represents the PC for 
%                       subject 'j' that belongs to synergy 'i'.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          21/07/21
% LAST MODIFIED:    21/07/21

number_of_cluster = size(pcs,1) / (number_of_subjects);

if number_of_cluster == 1 % Base case
    new_synergy = [synergies; {pcs}];
else

    [idx_means,C,sumd,D] = kmeans(pcs,number_of_cluster, 'Distance', 'sqEuclidean','EmptyAction','error','OnlinePhase','on','Replicates',25);
    aux_kmean = reshape(idx_means, [number_of_cluster, number_of_subjects]); % Split by subjects

    eva = evalclusters(pcs, idx_means, 'silhouette'); % Evaluate clusters
    [silh3,h] = silhouette(pcs,idx_means,'sqEuclidean');
    xlabel('Silhouette Value')
    ylabel('Cluster')

    [sorted_scores, original_pos] = max(eva.ClusterSilhouettes{1,1}(:)); % Get cluster with max score   
    [row_max, col_max] = find(ismember(aux_kmean,original_pos));

    if sorted_scores > 0 % The cluster has positive score

        % PC only once per subject and present in all subjects
        if numel(col_max) == numel(unique(col_max)) && numel(col_max) == number_of_subjects

            % Add PCs with max score
            aux_synergy = {pcs(ismember(idx_means,original_pos),:)}; 

            % Remove PCs added to the synergy
            pos_remove = ismember(idx_means,original_pos);
            new_pcs = pcs(~pos_remove,:);

        % PC only once per subject but NOT present in all subjects
        elseif numel(col_max) == numel(unique(col_max)) && numel(col_max) < number_of_subjects

        % PC repeated in some subject and present in all subjects
        elseif numel(col_max) > numel(unique(col_max)) && numel(unique(col_max)) == number_of_subjects

        % PC repeated in some subject but NOT present in all subjects
        elseif numel(col_max) > numel(unique(col_max)) && numel(unique(col_max)) < number_of_subjects

        end 

        % Save synergy
        % Recursive call (check before)

        % Add the extracted synergy
        new_synergy = [synergies; aux_synergy]; 

        % Recursive call if we have not finished
        if size(new_pcs,1) > 0
            synergies = recursive_clustering(new_pcs, number_of_subjects, new_synergy);
        end
    
    end
end

end