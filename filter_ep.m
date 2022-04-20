function [filtered_data, ep_labels, task_labels, time] = filter_ep(in_data, ep)
% FILTER_EP Function filter the data by a particular EP.
%
% INPUT
% in_data:          Cell array. Each position corresponds to the data for a
%                   particular subject in a MxN string array. Rows
%                   represent timepoints while columns represent joints,
%                   except last column which contains the EP. All columns
%                   are treated as strings.
%
% ep:               String representing the EP to select. In case 'ep' is 
%                   empty the function returns all datapoints.              
%
% OUTPUT
% filtered_data:    Cell array with same structure as 'in_data' containing
%                   the selected timepoints. Column containing the EP has
%                   been removed.
%
% ep_labels:        If 'ep' is empty, 'labels' is a cell array containing
%                   the labels for each timepoint. Otherwise, 'labels' is
%                   a string with the selected EP.
%
% task_labels:      
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          29/03/22
% LAST MODIFIED:    12/04/22

filtered_data = {};
task_labels = {};
time = {};

if ~isempty(ep)
    ep_labels = string(ep);
else
    ep_labels= {};
end

for iter = 1:numel(in_data)
   
    subj_data = in_data{iter}; 
    
    if ~isempty(ep)
        idx = matches(subj_data(:,end),ep);
        selected_data = subj_data(idx,1:end-3);
    else
        selected_data = subj_data(:,1:end-3);
        ep_labels{end+1} = subj_data(:,end-2);
    end
        
    filtered_data{end+1} = str2double(selected_data);
    task_labels{end+1} = subj_data(:,end-1);
    time{end+1} = str2double(subj_data(:,end));

end

end