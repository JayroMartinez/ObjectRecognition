function handplot_movement_reconstruction(mean_pos, synergies, scores, selected_syn)
% mean_pos = hand mean position for that particular subject
% synergies = all synergies for that particular subject
% scores = scores for the selected synergies (activation)
% selected_syn = array with those synergies we have selected


number_of_plots = 5;
synergies = squeeze(synergies);
starting_conf = deg2rad(cat(2,mean_pos(19), mean_pos(1), mean_pos(2), -mean_pos(3), mean_pos(15), mean_pos(4), mean_pos(5), 0, mean_pos(18), mean_pos(6), mean_pos(7), 0, mean_pos(17), mean_pos(8), mean_pos(9), 0, mean_pos(16), mean_pos(10), mean_pos(11), 0));
empty = zeros(1, size(synergies,1));
hand_syn = [
            synergies(19,:),   % Thumb Ab
            synergies(1,:),    % Thumb Rot
            synergies(2,:),    % Thumb MPJ Flex
            -synergies(3,:),   % Thumb  Ij
            synergies(15,:),   % Index MPJ Ab
            synergies(4,:),    % Index MPJ Flex
            synergies(5,:),    % Index PIJ
            empty,             % Index DIP       [EMPTY]
            synergies(18,:),   % Middle MPJ Ab
            synergies(6,:),    % Middle MPJ Flex
            synergies(7,:),    % Middle PIJ
            empty,             % Middle DIP      [EMPTY]
            synergies(17,:),   % Ring MPJ Ab
            synergies(8,:),    % Ring MPJ Flex
            synergies(9,:),    % Ring PIJ
            empty,             % Ring DIP        [EMPTY]
            synergies(16,:),   % Pinkie MPJ Ab
            synergies(10,:),   % Pinkie MPJ Flex
            synergies(11,:),   % Pinkie PIJ
            empty];            % Pinkie DIP      [EMPTY]

hand_syn(isnan(hand_syn)) = 0;
        
new_scores = zeros(size(scores,1), size(synergies,2));
for iter = 1:numel(selected_syn)
    
    new_scores(:,selected_syn(iter)) = scores(:,iter);
    
end

new_scores(isnan(new_scores)) = 0;

timepoints_to_plot = ceil(linspace(1, size(new_scores, 1),number_of_plots));
selected_activations = new_scores(timepoints_to_plot, :);


ha = SGparadigmatic; % Define and create hand
ha.S = hand_syn;

% figure;
% hand = SGmoveHand(ha, starting_conf');
% SGplotHand(hand);
figure;
for it = 1:size(selected_activations,1)
    
    hand = SGmoveHand(ha, starting_conf' + ha.S * selected_activations(it,:)');
    subplot(1, size(selected_activations,1), it);
    SGplotHand(hand);
    
end

a=1;








end