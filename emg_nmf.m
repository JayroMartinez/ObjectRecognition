function emg_nmf(filtered)
% THIS FUNCTION COMPUTES NON-NEGATIVE MATRIX FACTORIZATION ON THE EMG
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          30/06/22
% LAST MODIFIED:    30/06/22

% PUT ALL SUBJECTS DATA TOGETHER 
all_subjects = [];

for k = 1:numel(filtered)

    [all_subjects] = [all_subjects; filtered{k}];

end

clc;
init_time = tic;

num_syn_extract = [4,8,12,16,24,32,48,64];

% LOOP OVER NUMBER OF SYNERGYES TO EXTRACT FROM NMF
for i=1:length(num_syn_extract)
    
    loop_time = tic;
    
    [W,H] = nnmf(all_subjects, num_syn_extract(i));
    disp(['Elapsed time for num syn = ' num2str(num_syn_extract(i)) ': ' num2str(toc(loop_time)) ' seconds']);
    
    path = [pwd '/PyCode/PyData'];
    emg_w = jsonencode(W);
    emg_h = jsonencode(H);
    w_path = fullfile(path, ['W_', num2str(num_syn_extract(i)) '.json']);
    h_path = fullfile(path, ['H_' num2str(num_syn_extract(i)) '.json']);
    w_file = fopen(w_path, 'w');
    h_file = fopen(h_path, 'w');
    fprintf(w_file, emg_w);
    fprintf(h_file, emg_h);
    clear emg_w emg_h w_path h_path w_file h_file

end


disp(['Total Elapsed time: ' num2str(toc(init_time)) ' seconds']);

end