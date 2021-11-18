function [out_data] = load_subject_glove(subject_to_load)

% LOAD_SUBJECT_GLOVE Function to load data corresponding to a single subject.
%
% INPUT
% subject_to_load:  String with the subject's name.
%
% OUTPUT
% out_data:         Cyberglove data from the corresponding subject.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          24/06/21
% LAST MODIFIED:    18/11/21

% file = [pwd,'/Data/', 'Data_', subject_to_load, '.mat'];
file = [pwd,'/Data/', 'Cut_Data_', subject_to_load, '.mat'];

aux_data = load(file);

out_data = [];

for i = 1:numel(aux_data.haptic_exploration_data.subjects.tasks)

    out_data = [out_data; aux_data.haptic_exploration_data.subjects.tasks(i).data(4).data];
   
end


end