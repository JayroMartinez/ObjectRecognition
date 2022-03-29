function synergies = clustering(pcs, number_of_subjects)

% CLUSTERING Generates the synergies through clustering. It should be
% notices that the clusters have no restrictions so could be clusters with
% more that ine PC per subject, subjects with some PC that is not present
% in any cluster and so on.
%
% INPUT
% pcs:                  Matrix PCA coefficients. Each row represents a PC 
%                       from and each column represents a joint. Notice 
%                       that the number of rows is [number of subjects X 
%                       PCs per subject]. After last PC from a subject it 
%                       comes the first PC of the next subject. The last 
%                       group of PCs correspond to those calculated from 
%                       the data for all subjects together (in case we 
%                       selected to include it).
%
% number_of_subjects:   Number of subjects that has been loaded.
%
% OUTPUT
% synergies:            Array with the synergies. Each value in position 
%                       (i,j) represents the PC for subject 'j' that 
%                       belongs to synergy 'i'.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          19/07/21
% LAST MODIFIED:    22/07/21

number_of_synergies = size(pcs,1) / (number_of_subjects);

% [idx_means,C,sumd,D] = kmeans(pcs,number_of_synergies, 'Distance', 'sqEuclidean','EmptyAction','error','OnlinePhase','on','Replicates',250);
[idx_means,C,sumd,D] = kmeans(pcs,number_of_synergies, 'Distance', 'cosine','EmptyAction','error','OnlinePhase','on','Replicates',250);
aux_kmean = reshape(idx_means, [number_of_synergies, number_of_subjects]);

[silh3,h] = silhouette(pcs,idx_means,'cosine');
xlabel('Silhouette Value');
ylabel('Cluster');
close all;
clusters = aux_kmean;

%% Synergy creation

synergies = [];

for iter = 1:size(clusters,1)
    
    % Find all PCs corresponding to cluster 'i'
    [row_i, col_i] = find(ismember(aux_kmean,iter));
    
    aux_synergy = [];

    % PC only once per subject and present in all subjects
    if numel(col_i) == numel(unique(col_i)) && numel(col_i) == number_of_subjects

        for i = 1:number_of_subjects
            aux_synergy = [aux_synergy, row_i(find(ismember(col_i,i)))];
        end

    % PC only once per subject but NOT present in all subjects
    elseif numel(col_i) == numel(unique(col_i)) && numel(col_i) < number_of_subjects

        for i = 1:number_of_subjects
            if ~isempty(find(ismember(col_i,i))) % If subject is present in the synergy

                aux_synergy = [aux_synergy, row_i(find(ismember(col_i,i)))];

            else % Else, write NaN

                aux_synergy = [aux_synergy, NaN];

            end
        end

    % PC repeated in some subject and present in all subjects
    elseif numel(col_i) > numel(unique(col_i)) && numel(unique(col_i)) == number_of_subjects

        clu_sc = reshape(silh3, [], number_of_subjects); % PC silhouette scores by subject

        for i = 1:number_of_subjects

            if numel(find(ismember(col_i,i))) == 1 % If there is only one PC for subject 'i'

                aux_synergy = [aux_synergy, row_i(find(ismember(col_i,i)))];

            else

                % Get sihlouette scores for 'i' subject's PCs
                sh_max = max(clu_sc(row_i(find(ismember(col_i,i))),i));

                % Get PC with higher silhouette value
                sh_max_pos = find(ismember(clu_sc(:,i),sh_max));

                % Save that PC
                aux_synergy = [aux_synergy, sh_max_pos];

            end

        end
        
    % PC repeated in some subject but NOT present in all subjects
    elseif numel(col_i) > numel(unique(col_i)) && numel(unique(col_i)) < number_of_subjects

        clu_sc = reshape(silh3, [], number_of_subjects); % PC silhouette scores by subject

        for i = 1:number_of_subjects

            if isempty(find(ismember(col_i,i))) % If subject is NOT present in the synergy
                
                aux_synergy = [aux_synergy, NaN];
            
            elseif numel(find(ismember(col_i,i))) == 1 % If there is only one PC for subject 'i'

                aux_synergy = [aux_synergy, row_i(find(ismember(col_i,i)))];

            else % If there is more than one PC for subject 'i'

                % Get sihlouette scores for 'i' subject's PCs
                sh_max = max(clu_sc(row_i(find(ismember(col_i,i))),i));

                % Get PC with higher silhouette value
                sh_max_pos = find(ismember(clu_sc(:,i),sh_max));

                % Save that PC
                aux_synergy = [aux_synergy, sh_max_pos];

            end

        end

    end
    
    synergies = [synergies; aux_synergy];
    
end

end