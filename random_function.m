function random_function(sorted_syn, pca_values, means, stdevs, joint_names, subjects)
%% SYNERGY CALCULATION

% suj_scores = cell2mat(pca_values(1,2));
% 
% means_suj3 = means(1,:);
% stdevs_suj3 = stdevs(1,:);
% 
% pc_points = suj_scores(:,1);
% 
% % Syngrasp is made to use Z-scored coordinates and not absolute coordinates
% % max_pc = max(pc_points) * stdevs_suj3(1) + means_suj3(1); % point * std + mean 
% % min_pc = min(pc_points) * stdevs_suj3(1) + means_suj3(1); 
% max_pc = max(pc_points) + means_suj3(1); % point * std + mean 
% min_pc = min(pc_points) + means_suj3(1); 
% 
% interpolated_points = min_pc:(range([min_pc, max_pc])/2):max_pc;
% interpolated_points(3) = 0;
% 
% coeffs = cell2mat(pca_values(1,1));
% pc1 = coeffs(:,1)';
% 
% % NxM matrix, each row is an observation (point), each column is an axis
% % (old variables)
% rec = interpolated_points' * pc1;

%% HAND PLOTS ?????
% Once we have calculated th synergies, we are going to plot the hand, or
% at least, try it

% pc_to_plot = rec(5,:);
% 
% hand_conf = [pc_to_plot(1), % Thumb Rot
%             pc_to_plot(4), % Thumb Ab
%             pc_to_plot(2), % Thumb MPJ F
%             pc_to_plot(3), % Thumb Thumb Ij
%             0, % Index MPJ Ab
%             pc_to_plot(5), % Index MPJ F
%             pc_to_plot(6), % Index PIJ
%             0, % Index DIP
%             0, % Middle MPJ Ab
%             pc_to_plot(7), % Middle MPJ F
%             pc_to_plot(8), % Middle PIJ
%             0, % Middle DIP
%             0, % Ring MPJ Ab
%             pc_to_plot(10), % Ring MPJ F
%             pc_to_plot(11), % Ring PIJ
%             0, % Ring DIP
%             0, % Little MPJ Ab
%             pc_to_plot(13), % Little MPJ F
%             pc_to_plot(14), % Little PIJ
%             0]; % Little DIP     
% 
% ha = SGparadigmatic; % Define and create hand
% ha = SGmoveHand(ha, hand_conf);
% 
% close all;
% figure;
% SGplotHand(ha);
% % SGplotSyn(ha,z,qm);
% axis('equal');


%% DEFINE SYNERGIES

syn_to_plot = 10;

% load('qm.mat');        
% starting_conf = qm';

empty_conf = zeros(size(means,1),1);
starting_conf = deg2rad(cat(2,means(:,1), means(:,4), means(:,2), means(:,3), empty_conf, means(:,5), means(:,6), empty_conf, empty_conf, means(:,7), means(:,8), empty_conf, empty_conf, means(:,10), means(:,11), empty_conf, empty_conf, means(:,13), means(:,14), empty_conf));

hand_array = [];

original_subjects = subjects;

for suj = 1:size(sorted_syn,2)
    
    get_pc = sorted_syn(syn_to_plot, suj);
    
    if ~isnan(get_pc)
    
        coeffs = cell2mat(pca_values(suj,1));

        empty_row = zeros(1,size(coeffs,1));

        hand_syn = [coeffs(1,:),    % Thumb Rot
                    coeffs(4,:),    % Thumb Ab
                    coeffs(2,:),    % Thumb MPJ F
                    coeffs(3,:),    % Thumb Thumb Ij
                    empty_row,      % Index MPJ Ab   [EMPTY]
                    coeffs(5,:),    % Index MPJ F
                    coeffs(6,:),    % Index PIJ
                    empty_row,      % Index DIP      [EMPTY]
                    empty_row,      % Middle MPJ Ab  [EMPTY]
                    coeffs(7,:),    % Middle MPJ F
                    coeffs(8,:),    % Middle PIJ
                    empty_row,      % Middle DIP     [EMPTY]
                    empty_row,      % Ring MPJ Ab    [EMPTY]
                    coeffs(10,:),   % Ring MPJ F
                    coeffs(11,:),   % Ring PIJ
                    empty_row,      % Ring DIP       [EMPTY]
                    empty_row,      % Little MPJ Ab  [EMPTY]
                    coeffs(13,:),   % Little MPJ F
                    coeffs(14,:),   % Little PIJ
                    empty_row];     % Little DIP     [EMPTY]

        synergy_activation = zeros(size(coeffs,1),1);
        synergy_activation(get_pc) = 1;


        ha = SGparadigmatic; % Define and create hand
        ha.S = hand_syn;

        ha_init = SGmoveHand(ha, starting_conf(suj,:)' + ha.S * synergy_activation);
        ha_mean = SGmoveHand(ha, starting_conf(suj,:)');
        ha_end = SGmoveHand(ha, starting_conf(suj,:)' - ha.S * synergy_activation);

        hand_array = [hand_array; ha_init ha_mean ha_end];
    
    else
        subjects(strcmp(subjects,original_subjects(suj))) = []
    end
    
end



subjects = strrep(subjects, '_', ' ');
positions = {'Initial';'Mean';'Final'};

for i = 1:size(hand_array,1)
    
    plot_name = ['Synergy ' num2str(syn_to_plot)];
    figure('Name',plot_name);

    for j = 1:size(hand_array,2)
    
        subplot(1, size(hand_array,2), j);
        ttl = [subjects{i} ' ' positions{j}];
        SGplotHand(hand_array(i,j));
        title(string(ttl));
        axis('equal');
        
    end
end

end