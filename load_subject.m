function [out_data] = load_subject(subject_to_load)

% LOADSUBJECT Function to load data corresponding to a single subject.
%
% INPUT
% subject_to_load:  String with the subject's name.
%
% OUTPUT
% out_data:         Data from the corresponding subject.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          24/06/21
% LAST MODIFIED:    24/06/21

path = fullfile(pwd,'Data',subject_to_load,'haptic_exploration');

old_dir = pwd;
d = dir(path); 
isub = [d(:).isdir]; 
nameFolds = string({d(isub).name});
nameFolds = nameFolds(nameFolds ~= '.');
nameFolds = nameFolds(nameFolds ~= '..');
cd(old_dir);

folders_to_exclude = {'calibration', 'Calibration', 'CALIBRATION'};

subjects_data = [];

for i = 1:numel(nameFolds)
    
    if ~ismember(nameFolds(i),folders_to_exclude);
        
        file = fullfile(path,nameFolds(i),'_cyberglove_calibrated_joint_states.csv');
    
        aux_data = readtable(file);
        subjects_data = [subjects_data; aux_data];
        
        clear aux_data;
    end
end

out_data = subjects_data;

clear subjects_data;

end