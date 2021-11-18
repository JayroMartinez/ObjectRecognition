function [out_data] = load_subject(subject_to_load, sources)

% LOAD_SUBJECT Function to load data corresponding to a single subject. The
%              aim of this function is to call the load functions
%              corresponding to the selected sources. If more that one
%              source is selected some trasnformation is required in order
%              to have same number of datapoints per trial on each source.
%
% INPUT
% subject_to_load:  String with the subject's name.
% sources:          Logical array containing the selected sources. Each
%                   position corresponds to a particular source. 
%                   [ Glove Vicon ].
%
% OUTPUT
% out_data:         Data from the corresponding subject.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          18/11/21
% LAST MODIFIED:    18/11/21


% Perform DTW over Glove & Vicon time to have both sources with the same
% number of datapoints and merge both signals in the same dataset.
if sources(1) & sources(2)
    glove_data = load_subject_glove(subject_to_load);
    vicon_data = load_subject_vicon(subject_to_load);
    
    % DTW
    
    % MERGE DATA
   
else
    if sources(1) % Glove
        glove_data = load_subject_glove(subject_to_load);
        % Remove the abduction/adduction variables
        fields_to_remove = {'ThumbAb', 'MiddleIndexAb', 'RingMiddleAb', 'PinkieRingAb'};
        out_data = removevars(glove_data, fields_to_remove);
        
    else sources(2) % Vicon
        vicon_data = load_subject_vicon(subject_to_load);
        % Select only Time and abduction/adduction variables
        fields_to_select = {'UNIX_time', 'Index_Proj_J1_Z', 'Pinkie_Proj_J1_Z', 'Ring_Proj_J1_Z', 'Middle_Proj_J1_Z', 'Thumb_Proj_J1_Z'};
        out_data = vicon_data(:, fields_to_select);
    end
end


end