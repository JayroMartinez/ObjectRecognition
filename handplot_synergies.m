function handplot_synergies(sorted_pcs, means, synergy_to_plot, subjects_to_load)

close all; 

selected_pcs = sorted_pcs(:,:,synergy_to_plot);

empty_conf = zeros(size(means,1),1);
starting_conf = deg2rad(cat(2,means(:,19), means(:,1), means(:,2), -means(:,3), means(:,15), means(:,4), means(:,5), empty_conf, means(:,18), means(:,6), means(:,7), empty_conf, means(:,17), means(:,8), means(:,9), empty_conf, means(:,16), means(:,10), means(:,11), empty_conf));

hand_array = [];

original_subjects = subjects_to_load;

for suj = 1:size(selected_pcs,2)
    
    if sum(isnan(selected_pcs(:,suj))) == 0
    
        coeffs = selected_pcs(:,suj);
        hand_syn = zeros(20, size(sorted_pcs,3));

        empty_row = zeros(1,size(coeffs,1));

        hand_syn(:,synergy_to_plot) = [
            coeffs(19),   % Thumb Ab
            coeffs(1),    % Thumb Rot
            coeffs(2),    % Thumb MPJ Flex
            -coeffs(3),   % Thumb Thumb Ij
            coeffs(15),   % Index MPJ Ab
            coeffs(4),    % Index MPJ Flex
            coeffs(5),    % Index PIJ
            0,            % Index DIP       [EMPTY]
            coeffs(18),   % Middle MPJ Ab
            coeffs(6),    % Middle MPJ Flex
            coeffs(7),    % Middle PIJ
            0,            % Middle DIP      [EMPTY]
            coeffs(17),   % Ring MPJ Ab
            coeffs(8),    % Ring MPJ Flex
            coeffs(9),    % Ring PIJ
            0,            % Ring DIP        [EMPTY]
            coeffs(16),   % Pinkie MPJ Ab
            coeffs(10),   % Pinkie MPJ Flex
            coeffs(11),   % Pinkie PIJ
            0];           % Pinkie DIP      [EMPTY]
        
        synergy_activation = zeros(size(coeffs,1),1);
        synergy_activation(synergy_to_plot) = 1.5;
        
        ha = SGparadigmatic; % Define and create hand
        ha.S = hand_syn;

        ha_init = SGmoveHand(ha, starting_conf(suj,:)' + ha.S * synergy_activation);
        ha_mean = SGmoveHand(ha, starting_conf(suj,:)');
        ha_end = SGmoveHand(ha, starting_conf(suj,:)' - ha.S * synergy_activation);

        hand_array = [hand_array; ha_init ha_mean ha_end];
    
    else
        subjects_to_load(strcmp(subjects_to_load,original_subjects(suj))) = [];
    end
    
end



subjects_to_load = strrep(subjects_to_load, '_', ' ');
positions = {'Initial';'Mean';'Final'};

for i = 1:size(hand_array,1)
    
    plot_name = ['Synergy ' num2str(synergy_to_plot)];
    figure('Name',plot_name);

    for j = 1:size(hand_array,2)
    
        subplot(1, size(hand_array,2), j);
        ttl = [subjects_to_load{i} ' ' positions{j}];
        SGplotHand(hand_array(i,j));
        title(string(ttl));
        axis('equal');
        
    end
end

end