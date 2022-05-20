function synergy_variances_oneTOone = syn_variance_calculation_oneTOone(sorted_pcs)

% SYN_VARIANCE_CALCULATION Function to load data corresponding to a single subject.
%
% INPUT
% sorted_pcs:   M x N x P couble array [Joints x Subjects x Synergies].
%               Value (m,n,p) correspond to the joint 'm' for subject 'n'
%               and synergy 'p'.
%
% OUTPUT
% synergy_variances: 
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          07/12/21
% LAST MODIFIED:    20/01/22


% synergy_variances = {};
% syn_var = NaN(size(sorted_pcs, 3),1);

distances = [];

for syn = 1:size(sorted_pcs, 3)
    
    aux_dist = [];
    
    for suj1 = 1:size(sorted_pcs, 2)
        
        if sum(isnan(sorted_pcs(:,suj1,syn))) == 0
        
            
            
            for suj2 = suj1+1:size(sorted_pcs, 2)
                
                if sum(isnan(sorted_pcs(:,suj2,syn))) == 0
                    
                    s1 = sorted_pcs(:,suj1,syn)';
                    s2 = sorted_pcs(:,suj2,syn)';
                    
%                     temp_dist = pdist2(s1,s2,'cosine');
%                     dist = 1 - abs(1-temp_dist);
%                     aux_dist = [aux_dist dist];
                    

                    temp_angle = subspace(s1',s2');
                    temp_dist = cos(temp_angle);
                    dist = 1 - temp_dist;
                    aux_dist = [aux_dist dist];
                    
                end

            end
            
        end
        
    end
    
    distances{end+1} =  aux_dist;
   
%     mean_syn = mean(sorted_pcs(:,:,syn_1), 2, 'omitnan');
%     clean_syns =  sorted_pcs(:,all(~isnan(sorted_pcs(:,:,syn_1))));
%     dist  = pdist2(clean_syns', mean_syn', 'cosine');
%     eucl = pdist2(clean_syns', mean_syn', 'seuclidean');
%     synergy_variances{syn_1} = 1 - abs(1-dist);
%     syn_var(syn_1) = mean(synergy_variances{syn_1});
%     mean_eucl(syn_1) = mean(eucl);

end

mat = NaN(size(sorted_pcs,3),nchoosek(size(sorted_pcs,2),2));

for iter1=1:19
   
    for iter2=1:numel(distances{iter1})
       
        aux = distances{iter1};
        mat(iter1,iter2) = aux(iter2);
        
    end
    
end

% boxplot(mat');
close all;
% boxchart(mat', 'whisker', 2);
boxplot(mat', 'whisker', 500);
xlabel('Synergies');
ylabel('Cosine Distance');

synergy_variances_oneTOone = distances;

% BOXPLOT CELLARRAY

% disp(syn_var);
% 
% [aux_max,~] = cellfun(@size, synergy_variances);
% max_subj = max(aux_max);
% 
% data_to_plot = NaN(max_subj, size(sorted_pcs, 3));
% 
% for iter = 1:size(data_to_plot,2)
% 
%    data_to_plot(1:numel(synergy_variances{iter}), iter) = synergy_variances{iter};
%     
% end
% 
% x = 1:length(data_to_plot);
% max_dat = max(data_to_plot);
% min_dat = min(data_to_plot);
% figure;
% plot(syn_var, 'r');
% hold on;
% plot(mean_eucl, 'k');
% patch([x fliplr(x)], [max_dat  fliplr(min_dat)], [0.3010 0.7450 0.9330],'FaceAlpha',.25)
% legend('Mean Cosine Distance', 'Mean Mahalanobis Distance', 'Cosine Distance Range', 'Location', 'best');
% title('Cosine Distance for each Synergy');

end