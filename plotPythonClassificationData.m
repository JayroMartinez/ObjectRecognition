function plotPythonClassificationData

close all;
%% LOAD DATA
ann_load = load('./aux_results/ann.mat');
glm_load = load('./aux_results/glm.mat');
logisticRegression_load = load('./aux_results/logisticRegression.mat');
svm_load = load('./aux_results/svm.mat');

ann = ann_load.ann;
glm = glm_load.glm;
logisticRegression = logisticRegression_load.logisticRegression;
svm = svm_load.svm;

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
title('Logistic Regression');

%% Support Vector Machine PLOT
figure;
plot(bins, svm, 'LineWidth', 2);
yline(33.33, '--k', 'LineWidth', 2);
legend('Mugs', 'Plates', 'Geometric', 'Cutlery', 'Ball', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Support Vector Machine');

%% Means & Stds
figure;
plot(bins, ann_mean, 'b', 'LineWidth', 2);
hold on;
plot(bins, svm_mean, 'r', 'LineWidth', 2);
plot(bins, log_mean, 'g', 'LineWidth', 2);
plot(bins, glm_mean, 'm', 'LineWidth', 2);

yline(33.33, '--k', 'LineWidth', 2);
legend('ANN', 'SVM', 'LogRegr', 'GLM', 'Chance Level');
xlabel('Number of Bins');
ylabel('Accuracy %');
ylim([0 100]);
title('Algorithm Comparison');

end