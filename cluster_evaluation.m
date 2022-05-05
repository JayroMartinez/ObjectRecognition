function quality = cluster_evaluation(sorted_synergies, pcs)

% CLUSTER_EVALUATION This function evaluates the cluster quality.
%
% INPUT
% sorted_synergies:     NxM matrix where each row represents a synergy and 
%                       each column represents a subject. Each position 
%                       (i,j)represents the PC for subject 'j'that belongs 
%                       to synergy 'i'. 
%
% pcs:                  Matrix withPCA coefficients. Each row represents a 
%                       PC from and each column represents a joint. Notice 
%                       that the number of rows is [number of subjects X 
%                       PCs per subject]. After last PC from a subject it 
%                       comes the first PC of the next subject. The last 
%                       group of PCs correspond to those calculated from 
%                       the data for all subjects together.
%
% OUTPUT
% quality:              Vector containing the evaluation score for each
%                       synergy (Value between -1 and 1). 
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          26/07/21
% LAST MODIFIED:    26/07/21

clusters = NaN([numel(sorted_synergies) 1]);

number_synergies = size(sorted_synergies,1);

for subj = 1:size(sorted_synergies,2)
   
    for syn = 1:number_synergies
        
        if ~isnan(sorted_synergies(syn,subj))
            
           cluster_number = sorted_synergies(syn,subj) + number_synergies * (subj - 1);
           clusters(cluster_number) = syn;
           
        end
    end
end

eva = evalclusters(pcs, clusters, 'silhouette');
figure;
[silh3,h] = silhouette(pcs,clusters,'sqEuclidean');
quality = eva.ClusterSilhouettes{1,1};
close all;

end