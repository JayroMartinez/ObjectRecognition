function variance_plots(subjects_to_load, pca_values, all_subjects_explained)

% VARIANCE_PLOTS    Function to plot the variance explained by the PCs. The
%                   data plotted here corresponds to each subject and to 
%                   the combination of all of them.
%
% INPUT
% subjects_to_load:         Cell array containing the ID for all subjects.
% pca_values:               Cell array containing the variance explained 
%                           for each PC for any subject.
% all_subjects_explained:   Array containing the variance explained for
%                           each PC in the data corresponding to all
%                           subjects together.
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          19/07/21
% LAST MODIFIED:    20/07/21

data_to_plot = [];

for i = 1:numel(subjects_to_load)
   
    data_to_plot = [data_to_plot, cell2mat(pca_values(i))];
    
end

data_to_plot = [data_to_plot, all_subjects_explained];

subjects_to_load(end+1) = {'All Subjects'};

figure;
hold on;
subplot(1,2,1);
b = bar(data_to_plot);
set(b, {'DisplayName'}, subjects_to_load);
ylim([0 100]);
legend('Location', 'best', 'Interpreter', 'none');
title('Variance explained for each PC');

subplot(1,2,2);
p = plot(cumsum(data_to_plot));
set(p, {'DisplayName'}, subjects_to_load);
yline(90, 'k', 'DisplayName', '90% Variance', 'LineWidth',2);
ylim([0 100]);
legend('Location', 'best', 'Interpreter', 'none');
title('Cumulative variance');

end