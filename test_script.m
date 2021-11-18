%% This script is to test whether we could apply DTW to raw data to align both (Vicon and Cyberglove) signals in time. In summary: NO.

clc;
clear all;
close all;

file = [pwd,'/Data/', 'Raw_Data_Subject_4', '.mat'];

data = load(file);

% Load times
vicon_time = table2array(data.haptic_exploration_data.subjects.tasks(1).data(3).data(:,1));
glove_time = table2array(data.haptic_exploration_data.subjects.tasks(1).data(1).data(:,1));

% Apply DTW to time
[dist, nvt, ngt] = dtw(vicon_time,glove_time);
new_vicon_time = vicon_time(nvt);
new_glove_time = glove_time(ngt);

figure('Name', 'Time');
subplot(2,1,1);
plot(vicon_time, 'b');
hold on;
plot(glove_time, 'r');
title('Original Time');
legend('Vicon','Glove','Location', 'best');
subplot(2,1,2);
plot(new_vicon_time, 'b');
hold on;
plot(new_glove_time, 'r');
title('DWT Time');
legend('Vicon','Glove','Location', 'best');

% Load Index Data First Task (Ceramic Mug vs. Ceramic Mug)
vicon_cermug_index = table2array(data.haptic_exploration_data.subjects.tasks(1).data(3).data(:,18));
glove_cermug_index = table2array(data.haptic_exploration_data.subjects.tasks(1).data(1).data(:,6));
% Calculate DTW over signals
[~, nvi, ngi] = dtw(vicon_cermug_index,glove_cermug_index);
new_vicon_index = vicon_cermug_index(nvi);
new_glove_index = glove_cermug_index(ngi);
% Plot original Index Data (Ceramic Mug vs. Ceramic Mug)
figure('Name', 'Index');
subplot(2,1,1);
plot(vicon_time, vicon_cermug_index,'b');
hold on;
plot(glove_time, glove_cermug_index, 'r');
title('Original Signals');
legend('Vicon','Glove','Location', 'best');
subplot(2,1,2);
plot(new_vicon_index, 'b');
hold on;
plot(new_glove_index, 'r');
title('DWT Signals');
legend('Vicon','Glove','Location', 'best');

% Load Middle Data First Task (Ceramic Mug vs. Ceramic Mug)
vicon_cermug_middle = table2array(data.haptic_exploration_data.subjects.tasks(1).data(3).data(:,27));
glove_cermug_middle = table2array(data.haptic_exploration_data.subjects.tasks(1).data(1).data(:,8));
% Calculate DTW over signals
[~, nvm, ngm] = dtw(vicon_cermug_middle, glove_cermug_middle);
new_vicon_middle = vicon_cermug_middle(nvm);
new_glove_middle = glove_cermug_middle(ngm);
% Plot original Middle Data (Ceramic Mug vs. Ceramic Mug)
figure('Name', 'Middle');
subplot(2,1,1);
plot(vicon_time, vicon_cermug_middle,'b');
hold on;
plot(glove_time, glove_cermug_middle, 'r');
title('Original Signals');
legend('Vicon','Glove','Location', 'best');
subplot(2,1,2);
plot(new_vicon_middle, 'b');
hold on;
plot(new_glove_middle, 'r');
title('DWT Signals');
legend('Vicon','Glove','Location', 'best');

complete_vicon = [vicon_time, vicon_cermug_index, vicon_cermug_middle]';
complete_glove = [glove_time, glove_cermug_index, glove_cermug_middle]';

[~, ncv, ncg] = dtw(complete_vicon, complete_glove);

% Plot original Index Data (Ceramic Mug vs. Ceramic Mug)
figure('Name', 'Combined Index');
subplot(2,1,1);
plot(vicon_time, vicon_cermug_index,'b');
hold on;
plot(glove_time, glove_cermug_index, 'r');
title('Original Signals');
legend('Vicon','Glove','Location', 'best');
subplot(2,1,2);
plot(complete_vicon(1,ncv), complete_vicon(2,ncv), 'b');
hold on;
plot(complete_glove(1,ncg), complete_glove(2,ncg), 'r');
title('DWT Signals');
legend('Vicon','Glove','Location', 'best');

% Plot original Middle Data (Ceramic Mug vs. Ceramic Mug)
figure('Name', 'Combined Middle');
subplot(2,1,1);
plot(vicon_time, vicon_cermug_middle,'b');
hold on;
plot(glove_time, glove_cermug_middle, 'r');
title('Original Signals');
legend('Vicon','Glove','Location', 'best');
subplot(2,1,2);
plot(complete_vicon(1,ncv), complete_vicon(3,ncv), 'b');
hold on;
plot(complete_glove(1,ncg), complete_glove(3,ncg), 'r');
title('DWT Signals');
legend('Vicon','Glove','Location', 'best');

a=1;