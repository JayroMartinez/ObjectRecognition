function handplot_movement_reconstruction(mean_pos, synergies, scores, selected_syn, info, raw_data)
% mean_pos = hand mean position for that particular subject
% synergies = all synergies for that particular subject
% scores = scores for the selected synergies (activation)
% selected_syn = array with those synergies we have selected
% info = information to display in each plot as title
% raw_data = original data from the same timepoints, just for comparative
%            reasons

% new_glove(:,1) = new_glove(:,1)-45;
% new_glove(:,2) = -new_glove(:,2)+45;
% new_glove(:,3) = -new_glove(:,3);
% new_glove(:,4) = -new_glove(:,4);

number_of_plots = 5;
synergies = squeeze(synergies);
starting_conf = deg2rad(cat(2,mean_pos(19)-45, -mean_pos(1)+90, -mean_pos(2), -mean_pos(3), mean_pos(15), mean_pos(4), mean_pos(5), mean_pos(5), mean_pos(18), mean_pos(6), mean_pos(7), mean_pos(7), mean_pos(17), mean_pos(8), mean_pos(9), mean_pos(9), mean_pos(16), mean_pos(10), mean_pos(11), mean_pos(11)));
% starting_conf = deg2rad(cat(2,mean_pos(19)-45, -mean_pos(1)+90, -mean_pos(2), -mean_pos(3), mean_pos(15), mean_pos(4), mean_pos(5), 0, mean_pos(18), mean_pos(6), mean_pos(7), 0, mean_pos(17), mean_pos(8), mean_pos(9), 0, mean_pos(16), mean_pos(10), mean_pos(11), 0));
empty = zeros(1, size(synergies,1));

% hand_syn = [
%             synergies(19,:),   % Thumb Ab
%             -synergies(1,:),    % Thumb Rot
%             -synergies(2,:),    % Thumb MPJ Flex
%             -synergies(3,:),   % Thumb  Ij
%             synergies(15,:),   % Index MPJ Ab
%             synergies(4,:),    % Index MPJ Flex
%             synergies(5,:),    % Index PIJ
%             -synergies(5,:),             % Index DIP       [EMPTY]
%             synergies(18,:),   % Middle MPJ Ab
%             synergies(6,:),    % Middle MPJ Flex
%             synergies(7,:),    % Middle PIJ
%             -synergies(7,:),             % Middle DIP      [EMPTY]
%             synergies(17,:),   % Ring MPJ Ab
%             synergies(8,:),    % Ring MPJ Flex
%             synergies(9,:),    % Ring PIJ
%             -synergies(9,:),             % Ring DIP        [EMPTY]
%             synergies(16,:),   % Pinkie MPJ Ab
%             synergies(10,:),   % Pinkie MPJ Flex
%             synergies(11,:),   % Pinkie PIJ
%             -synergies(11,:)];            % Pinkie DIP      [EMPTY]

hand_syn = [
            synergies(19,:),   % Thumb Ab
            -synergies(1,:),    % Thumb Rot
            -synergies(2,:),    % Thumb MPJ Flex
            -synergies(3,:),   % Thumb  Ij
            empty,   % Index MPJ Ab
            synergies(4,:),    % Index MPJ Flex
            synergies(5,:),    % Index PIJ
            synergies(5,:),             % Index DIP       [EMPTY]
            empty,   % Middle MPJ Ab
            synergies(6,:),    % Middle MPJ Flex
            synergies(7,:),    % Middle PIJ
            synergies(7,:),             % Middle DIP      [EMPTY]
            empty,   % Ring MPJ Ab
            synergies(8,:),    % Ring MPJ Flex
            synergies(9,:),    % Ring PIJ
            synergies(9,:),             % Ring DIP        [EMPTY]
            empty,   % Pinkie MPJ Ab
            synergies(10,:),   % Pinkie MPJ Flex
            synergies(11,:),   % Pinkie PIJ
            synergies(11,:)];            % Pinkie DIP      [EMPTY]

n_empty = zeros(size(raw_data,1),1);        
reordered_raw = horzcat(raw_data(:,19)-45,-raw_data(:,1)+90,-raw_data(:,2),-raw_data(:,3),raw_data(:,15),raw_data(:,4),raw_data(:,5),raw_data(:,5),raw_data(:,18),raw_data(:,6),raw_data(:,7),raw_data(:,7),raw_data(:,17),raw_data(:,8),raw_data(:,9),raw_data(:,9),raw_data(:,16),raw_data(:,10),raw_data(:,11),raw_data(:,11));            
% reordered_raw = horzcat(raw_data(:,19)-45,-raw_data(:,1)+90,-raw_data(:,2),-raw_data(:,3),raw_data(:,15),raw_data(:,4),raw_data(:,5),n_empty,raw_data(:,18),raw_data(:,6),raw_data(:,7),n_empty,raw_data(:,17),raw_data(:,8),raw_data(:,9),n_empty,raw_data(:,16),raw_data(:,10),raw_data(:,11),n_empty);            

hand_syn(isnan(hand_syn)) = 0;
        
new_scores = zeros(size(scores,1), size(synergies,2));

for iter = 1:numel(selected_syn)
    
    new_scores(:,selected_syn(iter)) = scores(:,iter);
    
end

if sum(isnan(new_scores),'all') > 0
    info = strcat(info, [newline 'INCLUDES NANs']);
end

new_scores(isnan(new_scores)) = 0;

timepoints_to_plot = ceil(linspace(1, size(new_scores, 1),number_of_plots));
selected_activations = new_scores(timepoints_to_plot, :);

selected_raw = reordered_raw(timepoints_to_plot,:);

ha = SGparadigmatic; % Define and create hand
ha.S = hand_syn;

to_plot = [];

% figure;
% hand = SGmoveHand(ha, starting_conf');
% SGplotHand(hand);
% figure('Name',info);
for it = 1:numel(timepoints_to_plot)
    
    hand = SGmoveHand(ha, starting_conf' + ha.S * selected_activations(it,:)');
%     subplot(1, numel(timepoints_to_plot), it);
%     SGplotHand(hand);
    to_plot = [to_plot hand];
    
end
% sgtitle(info);

new_info = [info(1:strfind(info, 'Number')-2) newline 'RAW DATA'];
if sum(isnan(selected_raw),'all') > 0
    new_info = strcat(new_info, [newline 'INCLUDES MISSING VALUES']);
end
selected_raw(isnan(selected_raw)) = 0;


ha_raw = SGparadigmatic;
% figure('Name',new_info);
new_hands = [];
for it2 = 1:numel(timepoints_to_plot)
    
    movement = deg2rad(selected_raw(it2,:));
    hand_raw = SGmoveHand(ha_raw, movement);
%     subplot(1, numel(timepoints_to_plot), it2);
%     SGplotHand(hand_raw);
    new_hands = [new_hands hand_raw];
end

to_plot = [to_plot;new_hands];
a=1;
% sgtitle(new_info);

figure('Name',info);
for pl_1 = 1:size(to_plot,1)
    for pl_2 = 1:size(to_plot,2)
   
        subplot(2, size(to_plot,2), (pl_1-1)*size(to_plot,2)+pl_2);
        SGplotHand(to_plot(pl_1, pl_2));
    
        if pl_1 == 1 && pl_2 == ceil(size(to_plot,2)/2) 
            title(info);
        elseif pl_1 == 2 && pl_2 == ceil(size(to_plot,2)/2) 
            title(new_info);
        end
    
    end 
end
% sgtitle(info);
    
end