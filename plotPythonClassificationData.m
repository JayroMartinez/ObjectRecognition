function plotPythonClassificationData

close all;
%% LOAD DATA
ann_load = load('./aux_results/ann.mat');
glm_load = load('./aux_results/glm.mat');
logisticRegression_load = load('./aux_results/logisticRegression.mat');
svm_load = load('./aux_results/svm.mat');

emg_load = load('./aux_results/emg.mat');
kin_emg_load = load('./aux_results/kin_emg.mat');

ann = ann_load.ann;
glm = glm_load.glm;
logisticRegression = logisticRegression_load.logisticRegression;
svm = svm_load.svm;
emg = emg_load.emg;
kin_emg = kin_emg_load.kin_emg;

clear *_load

bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
% families = {'Mugs'; 'Plates'; 'Geometric'; 'Cutlery'; 'Ball'};

%% COMPUTE MEANS AND STD
ann_mean = mean(ann);
ann_sd = std(ann);

glm_mean = mean(glm);
glm_sd = std(glm);

log_mean = mean(logisticRegression);
log_sd = std(logisticRegression);

svm_mean = mean(svm);
svm_sd = std(svm);

emg_mean = mean(emg);
emg_sd = std(emg);

kin_emg_mean = mean(kin_emg);
kin_emg_sd = std(kin_emg);

%% ANN PLOT
figure;
plot(bins, ann, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Artificial Neural Network');

%% GLM PLOT
figure;
plot(bins, glm, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Generalized Linear Model');

%% Logistic Regression PLOT
figure;
plot(bins, logisticRegression, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Raw Kinematics Logistic Regression');

%% Support Vector Machine PLOT
figure;
plot(bins, svm, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Support Vector Machine');

%% EMG PLOT
figure;
plot(bins, emg, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('EMG Logistic Regression');

%% EMG + KIN PLOT
figure;
plot(bins, kin_emg, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Raw Kinematics + EMG Logistic Regression');

%% Means & Stds
figure;
plot(bins, log_mean, 'b', 'LineWidth', 2);
hold on;
plot(bins, emg_mean, 'r', 'LineWidth', 2);
plot(bins, kin_emg_mean, 'g', 'LineWidth', 2);

yline(33.33, '--k', 'LineWidth', 2);
legend('Raw Kin', 'EMG', 'Raw Kin + EMG', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Source Comparison');

end