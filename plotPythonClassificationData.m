function plotPythonClassificationData

close all;
%% LOAD DATA
% ann_load = load('./aux_results/ann.mat');
% glm_load = load('./aux_results/glm.mat');
% logisticRegression_load = load('./aux_results/logisticRegression.mat');
% svm_load = load('./aux_results/svm.mat');
% 
% emg_load = load('./aux_results/emg.mat');
% kin_emg_load = load('./aux_results/kin_emg.mat');
% kin_emg = kin_emg_load.kin_emg;

% KIN
kin02_load = load('./aux_results/kin_02.mat');
kin05_load = load('./aux_results/kin_05.mat');
kin1_load = load('./aux_results/kin_1.mat');
kin02 = kin02_load.kin_02;
kin05 = kin05_load.kin_05;
kin1 = kin1_load.kin_1;

% EMG
emg02_load = load('./aux_results/emg_02.mat');
emg05_load = load('./aux_results/emg_05.mat');
emg1_load = load('./aux_results/emg_1.mat');
emg125_load = load('./aux_results/emg_125.mat');
emg15_load = load('./aux_results/emg_15.mat');
emg02 = emg02_load.EMG_02;
emg05 = emg05_load.EMG_05;
emg1 = emg1_load.EMG_1;
emg125 = emg125_load.EMG_125;
emg15 = emg15_load.EMG_15;

% KIN + EMG
kin_emg02_load = load('./aux_results/kin_emg02.mat');
kin_emg05_load = load('./aux_results/kin_emg05.mat');
kin_emg1_load = load('./aux_results/kin_emg1.mat');
kin_emg125_load = load('./aux_results/kin_emg125.mat');
kin_emg02 = kin_emg02_load.kin_emg02;
kin_emg05 = kin_emg05_load.kin_emg05;
kin_emg1 = kin_emg1_load.kin_emg1;
kin_emg125 = kin_emg125_load.kin_emg125;
% NORMALIZED KIN + EMG
norm_kin_emg1_load = load('./aux_results/norm_kin_emg1.mat');
norm_kin_emg1 = norm_kin_emg1_load.norm_kin_emg1;

% NMF
nmf4_02_load = load('./aux_results/NMF4_02.mat');
nmf4_05_load = load('./aux_results/NMF4_05.mat');
nmf4_125_load = load('./aux_results/NMF4_125.mat');

nmf16_02_load = load('./aux_results/NMF16_02.mat');
nmf16_05_load = load('./aux_results/NMF16_05.mat');
nmf16_125_load = load('./aux_results/NMF16_125.mat');

nmf64_02_load = load('./aux_results/NMF64_02.mat');
nmf64_05_load = load('./aux_results/NMF64_05.mat');
nmf64_125_load = load('./aux_results/NMF64_125.mat');
nmf64_15_load = load('./aux_results/NMF64_15.mat');

nmf4_load = load('./aux_results/NMF4.mat');
nmf8_load = load('./aux_results/NMF8.mat');
nmf16_load = load('./aux_results/NMF16.mat');
nmf32_load = load('./aux_results/NMF32.mat');
nmf64_load = load('./aux_results/NMF64.mat');

nmf4_02 = nmf4_02_load.NMF4_02;
nmf4_05 = nmf4_05_load.NMF4_05;
nmf4_125 = nmf4_125_load.NMF4_125;

nmf16_02 = nmf16_02_load.NMF16_02;
nmf16_05 = nmf16_05_load.NMF16_05;
nmf16_125 = nmf16_125_load.NMF16_125;

nmf64_02 = nmf64_02_load.NMF64_02;
nmf64_05 = nmf64_05_load.NMF64_05;
nmf64_125 = nmf64_125_load.NMF64_125;
nmf64_15 = nmf64_15_load.NMF64_15;

nmf4 = nmf4_load.NMF4;
nmf8 = nmf8_load.NMF8;
nmf16 = nmf16_load.NMF16;
nmf32 = nmf32_load.NMF32;
nmf64 = nmf64_load.NMF64;

% Algorithm Comparison
% ann = ann_load.ann;
% glm = glm_load.glm;
% logisticRegression = logisticRegression_load.logisticRegression;
% svm = svm_load.svm;
% emg = emg_load.emg;


clear *_load

bins = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
% families = {'Mugs'; 'Plates'; 'Geometric'; 'Cutlery'; 'Ball'};

%% COMPUTE MEANS AND STD
% ann_mean = mean(ann);
% ann_sd = std(ann);
% 
% glm_mean = mean(glm);
% glm_sd = std(glm);
% 
% log_mean = mean(logisticRegression);
% log_sd = std(logisticRegression);
% 
% svm_mean = mean(svm);
% svm_sd = std(svm);
% 
% emg_mean = mean(emg);
% emg_sd = std(emg);

% kin_emg_mean = mean(kin_emg);
% kin_emg_sd = std(kin_emg);

kin02_mean = mean(kin02);
kin05_mean = mean(kin05);
kin1_mean = mean(kin1);

emg02_mean = mean(emg02);
emg05_mean = mean(emg05);
emg1_mean = mean(emg1);
emg125_mean = mean(emg125);
emg15_mean = mean(emg15);

kin_emg02_mean = mean(kin_emg02);
kin_emg05_mean = mean(kin_emg05);
kin_emg1_mean = mean(kin_emg1);
kin_emg125_mean = mean(kin_emg125);

norm_kin_emg1_mean = mean(norm_kin_emg1);

nmf4_mean = mean(nmf4);
nmf8_mean = mean(nmf8);
nmf16_mean = mean(nmf16);
nmf32_mean = mean(nmf32);
nmf64_mean = mean(nmf64);

nmf4_02_mean = mean(nmf4_02);
nmf4_05_mean = mean(nmf4_05);
nmf4_125_mean = mean(nmf4_125);

nmf16_02_mean = mean(nmf16_02);
nmf16_05_mean = mean(nmf16_05);
nmf16_125_mean = mean(nmf16_125);

nmf64_02_mean = mean(nmf64_02);
nmf64_05_mean = mean(nmf64_05);
nmf64_125_mean = mean(nmf64_125);
nmf64_15_mean = mean(nmf64_15);

%% ANN PLOT
% figure;
% plot(bins, ann, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Artificial Neural Network');

%% GLM PLOT
% figure;
% plot(bins, glm, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Generalized Linear Model');

%% Logistic Regression PLOT
% figure;
% plot(bins, logisticRegression, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics Logistic Regression');

%% Support Vector Machine PLOT
% figure;
% plot(bins, svm, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Support Vector Machine');

%% KIN PLOTS
% figure;
% plot(bins, kin02, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics Logistic Regression C = 0.2');
% 
% figure;
% plot(bins, kin05, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics Logistic Regression C = 0.5');
% 
% figure;
% plot(bins, kin1, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics Logistic Regression C = 1');

%% KIN Comparison
% figure;
% plot(bins, kin02_mean, 'g', 'LineWidth', 2);
% hold on;
% plot(bins, kin05_mean, 'y', 'LineWidth', 2);
% plot(bins, kin1_mean, 'b', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('C = 0.2', 'C = 0.5', 'C = 1', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG Model Comparison');

%% EMG PLOTS
% figure;
% plot(bins, emg02, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG Logistic Regression C = 0.2');
% 
% figure;
% plot(bins, emg1, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG Logistic Regression C = 1');
% 
% figure;
% plot(bins, emg15, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG Logistic Regression C = 1.5');

%% EMG Comparison
% figure;
% plot(bins, emg02_mean, 'g', 'LineWidth', 2);
% hold on;
% plot(bins, emg05_mean, 'y', 'LineWidth', 2);
% plot(bins, emg1_mean, 'b', 'LineWidth', 2);
% plot(bins, emg125_mean, 'r', 'LineWidth', 2);
% plot(bins, emg15_mean, 'c', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('C = 0.2', 'C = 0.5', 'C = 1', 'C = 1.25', 'C = 1.5', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG Model Comparison');

%% EMG + KIN PLOTS
% figure;
% plot(bins, kin_emg02, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics + EMG Logistic Regression C = 0.2');
% 
% figure;
% plot(bins, kin_emg125, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics + EMG Logistic Regression C = 1.25');

%% EMG + KIN Comparison
% figure;
% plot(bins, kin_emg02_mean, 'g', 'LineWidth', 2);
% hold on;
% plot(bins, kin_emg05_mean, 'y', 'LineWidth', 2);
% plot(bins, kin_emg1_mean, 'b', 'LineWidth', 2);
% plot(bins, kin_emg125_mean, 'r', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('C = 0.2', 'C = 0.5', 'C = 1', 'C = 1.25', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics + EMG Model Comparison');

%% EMG + KIN NORM Comparison
% figure;
% plot(bins, kin_emg1_mean, 'b', 'LineWidth', 2);
% hold on;
% plot(bins, norm_kin_emg1_mean, 'r', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Raw Signal', 'Normalized Signal', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Raw Kinematics + EMG Model Comparison');

%% NMF 4 PLOT
% figure;
% plot(bins, nmf4_02, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 4 syns C = 0.2');
% 
% figure;
% plot(bins, nmf4_05, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 4 syns C = 0.5');
% 
% figure;
% plot(bins, nmf4_125, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 4 syns C = 1.25');

% figure;
% plot(bins, nmf4_02_mean, 'b', 'LineWidth', 2);
% hold on;
% plot(bins, nmf4_05_mean, 'r', 'LineWidth', 2);
% plot(bins, nmf4_125_mean, 'g', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('C = 0.2', 'C = 0.5', 'C = 1.25', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF 4 syns Model Comparison');

%% NMF 16 PLOT
% figure;
% plot(bins, nmf16_02, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 16 syns C = 0.2');
% 
% figure;
% plot(bins, nmf16_05, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 16 syns C = 0.5');
% 
% figure;
% plot(bins, nmf16_125, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 16 syns C = 1.25');
% 
% figure;
% plot(bins, nmf16_02_mean, 'b', 'LineWidth', 2);
% hold on;
% plot(bins, nmf16_05_mean, 'r', 'LineWidth', 2);
% plot(bins, nmf16_125_mean, 'g', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('C = 0.2', 'C = 0.5', 'C = 1.25', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF 16 syns Model Comparison');

%% NMF 16 PLOT
% figure;
% plot(bins, nmf16, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 16 syns');

%% NMF 32 PLOT
% figure;
% plot(bins, nmf32, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 32 syns');

%% NMF 64 PLOT
% figure;
% plot(bins, nmf64_02, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 64 syns C = 0.2');
% 
% figure;
% plot(bins, nmf64_05, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 64 syns C = 0.5');
% 
% figure;
% plot(bins, nmf64_125, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 64 syns C = 1.25');
% 
% figure;
% plot(bins, nmf64_15, 'LineWidth', 2);
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF with 64 syns C = 1.5');
% 
% figure;
% plot(bins, nmf64_02_mean, 'b', 'LineWidth', 2);
% hold on;
% plot(bins, nmf64_05_mean, 'r', 'LineWidth', 2);
% plot(bins, nmf64_125_mean, 'g', 'LineWidth', 2);
% plot(bins, nmf64_15_mean, 'y', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('C = 0.2', 'C = 0.5', 'C = 1.25', 'C = 1.5', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('EMG NMF 64 syns Model Comparison');

%% NMF Comparison
% figure;
% plot(bins, nmf4_05_mean, 'b', 'LineWidth', 2);
% hold on;
% plot(bins, nmf16_125_mean, 'g', 'LineWidth', 2);
% plot(bins, nmf64_125_mean, 'r', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('NMF 4 C = 0.5', 'NMF 16 C = 1.25', 'NMF 64 C = 1.25', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('NMF Comparison');

%% Source Comparison
% figure;
% plot(bins, kin02_mean, 'b', 'LineWidth', 2);
% hold on;
% plot(bins, emg125_mean, 'k', 'LineWidth', 2);
% plot(bins, kin_emg1_mean, 'g', 'LineWidth', 2);
% plot(bins, norm_kin_emg1_mean, 'r', 'LineWidth', 2);
% plot(bins, nmf64_125_mean, 'c', 'LineWidth', 2);
% 
% yline(33.33, '--k', 'LineWidth', 2);
% legend('Raw Kin', 'EMG', 'Raw Kin + EMG', 'Normalized Raw Kin + EMG', 'NMF 64 syns', 'Chance Level');
% xlabel('Number of Bins');
% ylabel('Accuracy %');
% ylim([0 100]);
% title('Source Comparison');

end