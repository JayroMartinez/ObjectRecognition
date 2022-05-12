function handplot_movie(mean_pos, synergies, scores, selected_syn, info, raw_data)

pause(10);

synergies = squeeze(synergies);
starting_conf = deg2rad(cat(2,mean_pos(19)-45, -mean_pos(1)+90, -mean_pos(2), -mean_pos(3), mean_pos(15), mean_pos(4), mean_pos(5), 0, mean_pos(18), mean_pos(6), mean_pos(7), 0, mean_pos(17), mean_pos(8), mean_pos(9), 0, mean_pos(16), mean_pos(10), mean_pos(11), 0));
empty = zeros(1, size(synergies,1));

hand_syn = [
            synergies(19,:),   % Thumb Ab
            -synergies(1,:),    % Thumb Rot
            -synergies(2,:),    % Thumb MPJ Flex
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

n_empty = zeros(size(raw_data,1),1);        
reordered_raw = horzcat(raw_data(:,19)-45,-raw_data(:,1)+90,-raw_data(:,2),-raw_data(:,3),raw_data(:,15),raw_data(:,4),raw_data(:,5),n_empty,raw_data(:,18),raw_data(:,6),raw_data(:,7),n_empty,raw_data(:,17),raw_data(:,8),raw_data(:,9),n_empty,raw_data(:,16),raw_data(:,10),raw_data(:,11),n_empty);            

hand_syn(isnan(hand_syn)) = 0;
        
new_scores = zeros(size(scores,1), size(synergies,2));

for iter = 1:numel(selected_syn)
    
    new_scores(:,selected_syn(iter)) = scores(:,iter);
    
end

if sum(isnan(new_scores),'all') > 0
    info = strcat(info, [newline 'INCLUDES NANs']);
end

new_scores(isnan(new_scores)) = 0;

ha_raw = SGparadigmatic;
ha_recon = SGparadigmatic;
ha_raw.S = hand_syn;

for to_plot = 1:5:size(new_scores,1)
   
    movement = deg2rad(reordered_raw(to_plot,:));
    hand_raw = SGmoveHand(ha_raw, movement);
    
    hand_rec = SGmoveHand(ha_raw, starting_conf' + ha_raw.S * new_scores(to_plot, :)');
    
    figure(1);
    set(gcf,'Position', [0 0 1440 900]);
    
    sgtitle(info);
    subplot(2, 2, 1);
    SGplotHand(hand_raw);
    title('RAW DATA');
    v1 = [90 -90 -45];
    view(v1);
    subplot(2, 2, 2);
    SGplotHand(hand_raw);
    title('RAW DATA');
    v2 = [-90 90 45];
    view(v2);
    
    subplot(2, 2, 3);
    SGplotHand(hand_rec);
    title('RECONSTRUCTION');
    v3 = [90 -90 -45];
    view(v3);
    subplot(2, 2, 4);
    SGplotHand(hand_rec);
    title('RECONSTRUCTION');
    v4 = [-90 90 45];
    view(v4);
    
end


end