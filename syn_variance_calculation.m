function synergy_variances = syn_variance_calculation(sorted_pcs)

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


synergy_variances = {};
mean_syn_var = NaN(size(sorted_pcs, 3),1);
mean_eucl = NaN(size(sorted_pcs, 3),1);

for syn = 1:size(sorted_pcs, 3)
   
    mean_syn = mean(sorted_pcs(:,:,syn), 2, 'omitnan');
    clean_syns =  sorted_pcs(:,all(~isnan(sorted_pcs(:,:,syn))));
    dist  = pdist2(clean_syns', mean_syn', 'cosine');
    eucl = pdist2(clean_syns', mean_syn', 'euclidean');
    synergy_variances{syn} = 1 - abs(1-dist);
    mean_syn_var(syn) = mean(synergy_variances{syn});
    mean_eucl(syn) = mean(eucl);

end

disp(mean_syn_var);

[aux_max,~] = cellfun(@size, synergy_variances);
max_subj = max(aux_max);

data_to_plot = NaN(max_subj, size(sorted_pcs, 3));

for iter = 1:size(data_to_plot,2)

   data_to_plot(1:numel(synergy_variances{iter}), iter) = synergy_variances{iter};
    
end

x = 1:length(data_to_plot);
max_dat = max(data_to_plot);
min_dat = min(data_to_plot);
figure;
plot(mean_syn_var, 'r');
hold on;
plot(mean_eucl, 'g');
patch([x fliplr(x)], [max_dat  fliplr(min_dat)], [0.3010 0.7450 0.9330],'FaceAlpha',.25)
legend('Mean Cosine Distance', 'Mean Euclidean Distance', 'Cosine Distance Range', 'Location', 'best');
title('Cosine Distance for each Synergy');

end