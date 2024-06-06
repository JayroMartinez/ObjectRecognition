function handplot_save_movie(synergy_index)

    % Mapping from original variable order to the SynGrasp framework order
    % Original order: ['ThumbRotate', 'ThumbMPJ', 'ThumbIj', 'IndexMPJ', 'IndexPIJ', 'MiddleMPJ', 'MiddlePIJ', 'RingMIJ', 'RingPIJ', 'PinkieMPJ', 'PinkiePIJ', 'PalmArch', 'WristPitch', 'WristYaw', 'Index_Proj_J1_Z', 'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z', 'Thumb_Proj_J1_Z']
    % Framework order: ['ThumbAb', 'ThumbRot', 'ThumbMPJFlex', 'ThumbIJFlex', 'IndexMPJAb', 'IndexMPJFlex', 'IndexPIJFlex', 'IndexDIPFlex', 'MiddleMPJAb', 'MiddleMPJFlex', 'MiddlePIJFlex', 'MiddleDIPFlex', 'RingMPJAb', 'RingMPJFlex', 'RingPIJFlex', 'RingDIPFlex', 'PinkieMPJAb', 'PinkieMPJFlex', 'PinkiePIJFlex', 'PinkieDIPFlex']

    % Folder containing the data files
    data_folder = './PyCode/AllPython/results/Syn'; % Update with your actual path
    
    % Load mean positions, synergies, and scales
    mean_pos = csvread(fullfile(data_folder, 'kin_mean.csv'), 1, 1);
    scales = csvread(fullfile(data_folder, 'kin_scale.csv'), 1, 1);
    synergies = csvread(fullfile(data_folder, '/synergies/kin_syns.csv'), 1, 1); % Skip the first row and column
    
    rescaled = 3 * scales / max(scales);

    % Extract the specific synergy
    selected_synergy = synergies(synergy_index + 1, :);
    
    % Reorder Synergy
    reordered_syn = [
        selected_synergy(19) * rescaled(19),   % ThumbAb          <-- Thumb_Proj_J1_Z
        selected_synergy(1) * rescaled(1),    % ThumbRot         <-- ThumbRotate
        selected_synergy(2) * rescaled(2),    % ThumbMPJFlex     <-- ThumbMPJ
        selected_synergy(3) * rescaled(3),    % ThumbIJFlex      <-- ThumbIj
        selected_synergy(15) * rescaled(15),   % IndexMPJAb       <-- Index_Proj_J1_Z
        selected_synergy(4) * rescaled(4),    % IndexMPJFlex     <-- IndexMPJ
        selected_synergy(5) * rescaled(5),    % IndexPIJFlex     <-- IndexPIJ
        0,                      % IndexDIPFlex     <-- EMPTY
        selected_synergy(18) * rescaled(18),   % MiddleMPJAb      <-- Middle_Proj_J1_Z
        selected_synergy(6) * rescaled(6),    % MiddleMPJFlex    <-- MiddleMPJ
        selected_synergy(7) * rescaled(7),    % MiddlePIJFlex    <-- MiddlePIJ
        0,                      % MiddleDIPFlex    <-- EMPTY
        selected_synergy(17) * rescaled(17),   % RingMPJAb        <-- Ring_Proj_J1_Z
        selected_synergy(8) * rescaled(8),    % RingMPJFlex      <-- RingMIJ
        selected_synergy(9) * rescaled(9),    % RingPIJFlex      <-- RingPIJ
        0,                      % RingDIPFlex      <-- EMPTY
        selected_synergy(16) * rescaled(16),   % PinkieMPJAb      <-- Pinkie_Proj_J1_Z
        selected_synergy(10) * rescaled(10),   % PinkieMPJFlex    <-- PinkieMPJ
        selected_synergy(11) * rescaled(11),   % PinkiePIJFlex    <-- PinkiePIJ
        0                       % PinkieDIPFlex    <-- EMPTY
    ];

    % Prepare the starting configuration
    starting_conf = deg2rad([
        mean_pos(19) - 45,   % ThumbAb          <-- Thumb_Proj_J1_Z
        -mean_pos(1) + 90,   % ThumbRot         <-- ThumbRotate
        -mean_pos(2),        % ThumbMPJFlex     <-- ThumbMPJ
        -mean_pos(3),        % ThumbIJFlex      <-- ThumbIj
        mean_pos(15),        % IndexMPJAb       <-- Index_Proj_J1_Z
        mean_pos(4),         % IndexMPJFlex     <-- IndexMPJ
        mean_pos(5),         % IndexPIJFlex     <-- IndexPIJ
        0,                   % IndexDIPFlex     <-- EMPTY
        mean_pos(18),        % MiddleMPJAb      <-- Middle_Proj_J1_Z
        mean_pos(6),         % MiddleMPJFlex    <-- MiddleMPJ
        mean_pos(7),         % MiddlePIJFlex    <-- MiddlePIJ
        0,                   % MiddleDIPFlex    <-- EMPTY
        mean_pos(17),        % RingMPJAb        <-- Ring_Proj_J1_Z
        mean_pos(8),         % RingMPJFlex      <-- RingMIJ
        mean_pos(9),         % RingPIJFlex      <-- RingPIJ
        0,                   % RingDIPFlex      <-- EMPTY
        mean_pos(16),        % PinkieMPJAb      <-- Pinkie_Proj_J1_Z
        mean_pos(10),        % PinkieMPJFlex    <-- PinkieMPJ
        mean_pos(11),        % PinkiePIJFlex    <-- PinkiePIJ
        0                    % PinkieDIPFlex    <-- EMPTY
    ]);

    % Initialize the hand model with the synergy matrix
    ha = SGparadigmatic;
    ha.S = reordered_syn;

    % Number of frames in the video
    num_frames = 50;

    % Create a video writer object
    video_filename = fullfile('./PyCode/AllPython/results/videos', ['TEST_synergy_', num2str(synergy_index), '_video.avi']);
    v = VideoWriter(video_filename);
    open(v);

    % Loop through the frames to create the video
    for frame = 1:num_frames
        % Interpolate the movement from one extreme to the other
        scale_factor = (frame - 1) / (num_frames - 1);
        movement = (starting_conf' - 0.5 * reordered_syn') + scale_factor * reordered_syn';
        movement = movement(:);  % Ensure movement is a column vector

        % Move the hand model according to the current movement
        hand = SGmoveHand(ha, movement);

        % Create an offscreen figure for plotting
        fig = figure('Visible', 'off', 'Position', [0, 0, 1440, 900]);
        sgtitle(['Synergy ' num2str(synergy_index)]);

        subplot(1, 3, 1);
        SGplotHand(hand);
        title('View 1');
        view([90, -90, -45]);

        subplot(1, 3, 2);
        SGplotHand(hand);
        title('View 2');
        view([-90, 90, 45]);

        subplot(1, 3, 3);
        SGplotHand(hand);
        title('View 3');
        view([0, 0, 90]);

        % Capture the frame and write it to the video
        frame_data = getframe(fig);
        writeVideo(v, frame_data);

        % Close the figure to free up memory
        close(fig);
    end

    % Loop through the frames in reverse order to create the closing movement
    for frame = num_frames:-1:1
        % Interpolate the movement from one extreme to the other
        scale_factor = (frame - 1) / (num_frames - 1);
        movement = (starting_conf' - 0.5 * reordered_syn') + scale_factor * reordered_syn';
        movement = movement(:);  % Ensure movement is a column vector

        % Move the hand model according to the current movement
        hand = SGmoveHand(ha, movement);

        % Create an offscreen figure for plotting
        fig = figure('Visible', 'off', 'Position', [0, 0, 1440, 900]);
        sgtitle(['Synergy ' num2str(synergy_index)]);

        subplot(1, 3, 1);
        SGplotHand(hand);
        title('View 1');
        view([90, -90, -45]);

        subplot(1, 3, 2);
        SGplotHand(hand);
        title('View 2');
        view([-90, 90, 45]);

        subplot(1, 3, 3);
        SGplotHand(hand);
        title('View 3');
        view([0, 0, 90]);

        % Capture the frame and write it to the video
        frame_data = getframe(fig);
        writeVideo(v, frame_data);

        % Close the figure to free up memory
        close(fig);
    end

    % Close the video writer
    close(v);
end