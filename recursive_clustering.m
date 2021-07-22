function [synergies] = recursive_clustering(pcs, number_of_subjects, synergies, number_of_clusters)

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
% LAST MODIFIED:    22/07/21

if number_of_clusters > 2 

    [idx_means,C,sumd,D] = kmeans(pcs,number_of_clusters, 'Distance', 'sqEuclidean','EmptyAction','error','OnlinePhase','on','Replicates',25);
    aux_kmean = reshape(idx_means, [], number_of_subjects); % Split by subjects

    eva = evalclusters(pcs, idx_means, 'silhouette'); % Evaluate clusters
    [silh3,h] = silhouette(pcs,idx_means,'sqEuclidean');
    close all; % Close silhouette figure

    [sorted_scores, original_pos] = max(eva.ClusterSilhouettes{1,1}(:)); % Get cluster with max score   
    [row_max, col_max] = find(ismember(aux_kmean,original_pos));
        
    aux_synergy = [];

    % PC only once per subject and present in all subjects
    if numel(col_max) == numel(unique(col_max)) && numel(col_max) == number_of_subjects

        for i = 1:number_of_subjects
            aux_synergy = [aux_synergy, row_max(find(ismember(col_max,i)))];
        end

        % Delete data from PCs added to synergies
        pcs(ismember(idx_means,original_pos),:) = NaN;

    % PC only once per subject but NOT present in all subjects
    elseif numel(col_max) == numel(unique(col_max)) && numel(col_max) < number_of_subjects

        for i = 1:number_of_subjects
            if ~isempty(find(ismember(col_max,i))) % If subject is present in the synergy

                aux_synergy = [aux_synergy, row_max(find(ismember(col_max,i)))];

            else % Else, write NaN

                aux_synergy = [aux_synergy, NaN];

            end
        end

        % Delete data from PCs added to synergies
        pcs(ismember(idx_means,original_pos),:) = NaN;

    % PC repeated in some subject and present in all subjects
    elseif numel(col_max) > numel(unique(col_max)) && numel(unique(col_max)) == number_of_subjects

        clu_sc = reshape(silh3, [], number_of_subjects); % PC silhouette scores by subject

        for i = 1:number_of_subjects

            if numel(find(ismember(col_max,i))) == 1 % If there is only one PC for subject 'i'

                aux_synergy = [aux_synergy, row_max(find(ismember(col_max,i)))];

            else

                % Get sihlouette scores for 'i' subject's PCs
                sh_max = max(clu_sc(row_max(find(ismember(col_max,i))),i));

                % Get PC with higher silhouette value
                sh_max_pos = find(ismember(clu_sc(:,i),sh_max));

                % Save that PC
                aux_synergy = [aux_synergy, sh_max_pos];

            end

        end

        pcs_per_subject = size(aux_kmean,1);

        for j = 1:numel(aux_synergy)
            row_to_delete = (j-1)*pcs_per_subject + aux_synergy(j); 
            % Delete data from PCs added to synergies
            pcs(row_to_delete,:) = NaN;
        end

    % PC repeated in some subject but NOT present in all subjects
    elseif numel(col_max) > numel(unique(col_max)) && numel(unique(col_max)) < number_of_subjects

        clu_sc = reshape(silh3, [], number_of_subjects); % PC silhouette scores by subject

        for i = 1:number_of_subjects

            if isempty(find(ismember(col_max,i))) % If subject is NOT present in the synergy
                
                aux_synergy = [aux_synergy, NaN];
            
            elseif numel(find(ismember(col_max,i))) == 1 % If there is only one PC for subject 'i'

                aux_synergy = [aux_synergy, row_max(find(ismember(col_max,i)))];

            else % If there is more than one PC for subject 'i'

                % Get sihlouette scores for 'i' subject's PCs
                sh_max = max(clu_sc(row_max(find(ismember(col_max,i))),i));

                % Get PC with higher silhouette value
                sh_max_pos = find(ismember(clu_sc(:,i),sh_max));

                % Save that PC
                aux_synergy = [aux_synergy, sh_max_pos];

            end

        end

        pcs_per_subject = size(aux_kmean,1);

        for j = 1:numel(aux_synergy)
            row_to_delete = (j-1)*pcs_per_subject + aux_synergy(j); 
            % Delete data from PCs added to synergies
            pcs(row_to_delete,:) = NaN;
        end

    end 

    % Add the extracted synergy
    new_synergies = [synergies; aux_synergy]; 

    % Recursive call if we have not finished
    if number_of_clusters > 2
        synergies = recursive_clustering(pcs, number_of_subjects, new_synergies, number_of_clusters - 1);
    end
    
else % 2 clusters is the base case (not possible to match PCs in a single cluster)
% The code is the same as before but without the recursive calling
    
    [idx_means,C,sumd,D] = kmeans(pcs,number_of_clusters, 'Distance', 'sqEuclidean','EmptyAction','error','OnlinePhase','on','Replicates',25);
    aux_kmean = reshape(idx_means, [], number_of_subjects); % Split by subjects

%     eva = evalclusters(pcs, idx_means, 'silhouette'); % Evaluate clusters
    [silh3,h] = silhouette(pcs,idx_means,'sqEuclidean');
    close all; % Close silhouette figure
    
    for k=1:2
       
        [row_k, col_k] = find(ismember(aux_kmean,k)); % Select the PCs that belong to the synergy
        
        aux_synergy = [];

        % PC only once per subject and present in all subjects
        if numel(col_k) == numel(unique(col_k)) && numel(col_k) == number_of_subjects

            for i = 1:number_of_subjects
                aux_synergy = [aux_synergy, row_k(find(ismember(col_k,i)))];
            end

            pcs_per_subject = size(aux_kmean,1);
            
            for j = 1:numel(aux_synergy)
                row_to_delete = (j-1)*pcs_per_subject + aux_synergy(j); 
                % Delete data from PCs added to synergies
                pcs(row_to_delete,:) = NaN;
            end

        % PC only once per subject but NOT present in all subjects
        elseif numel(col_k) == numel(unique(col_k)) && numel(col_k) < number_of_subjects

            for i = 1:number_of_subjects
                if ~isempty(find(ismember(col_k,i))) % If subject is present in the synergy

                    aux_synergy = [aux_synergy, row_k(find(ismember(col_k,i)))];

                else % Else, write NaN

                    aux_synergy = [aux_synergy, NaN];

                end
            end

            pcs_per_subject = size(aux_kmean,1);
            
            for j = 1:numel(aux_synergy)
                row_to_delete = (j-1)*pcs_per_subject + aux_synergy(j); 
                % Delete data from PCs added to synergies
                pcs(row_to_delete,:) = NaN;
            end

        % PC repeated in some subject and present in all subjects
        elseif numel(col_k) > numel(unique(col_k)) && numel(unique(col_k)) == number_of_subjects

            clu_sc = reshape(silh3, [], number_of_subjects); % PC silhouette scores by subject

            for i = 1:number_of_subjects

                if numel(find(ismember(col_k,i))) == 1 % If there is only one PC for subject 'i'

                    aux_synergy = [aux_synergy, row_k(find(ismember(col_k,i)))];

                else

                    % Get sihlouette scores for 'i' subject's PCs
                    sh_max = max(clu_sc(row_k(find(ismember(col_k,i))),i));

                    % Get PC with higher silhouette value
                    sh_max_pos = find(ismember(clu_sc(:,i),sh_max));

                    % Save that PC
                    aux_synergy = [aux_synergy, sh_max_pos];

                end

            end

            pcs_per_subject = size(aux_kmean,1);

            for j = 1:numel(aux_synergy)
                row_to_delete = (j-1)*pcs_per_subject + aux_synergy(j); 
                % Delete data from PCs added to synergies
                pcs(row_to_delete,:) = NaN;
            end

        % PC repeated in some subject but NOT present in all subjects
        elseif numel(col_k) > numel(unique(col_k)) && numel(unique(col_k)) < number_of_subjects

            clu_sc = reshape(silh3, [], number_of_subjects); % PC silhouette scores by subject

            for i = 1:number_of_subjects

                if isempty(find(ismember(col_k,i))) % If subject is NOT present in the synergy

                    aux_synergy = [aux_synergy, NaN];

                elseif numel(find(ismember(col_k,i))) == 1 % If there is only one PC for subject 'i'

                    aux_synergy = [aux_synergy, row_k(find(ismember(col_k,i)))];

                else % If there is more than one PC for subject 'i'

                    % Get sihlouette scores for 'i' subject's PCs
                    sh_max = max(clu_sc(row_k(find(ismember(col_k,i))),i));

                    % Get PC with higher silhouette value
                    sh_max_pos = find(ismember(clu_sc(:,i),sh_max));

                    % Save that PC
                    aux_synergy = [aux_synergy, sh_max_pos];

                end

            end

            pcs_per_subject = size(aux_kmean,1);

            for j = 1:numel(aux_synergy)
                row_to_delete = (j-1)*pcs_per_subject + aux_synergy(j); 
                % Delete data from PCs added to synergies
                pcs(row_to_delete,:) = NaN;
            end

        end 

        % Add the extracted synergy
        synergies = [synergies; aux_synergy];

    end

end

end