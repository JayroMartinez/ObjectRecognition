function evo_weight_ep_task(sorted_scores, sorted_pcs, means, ep_labels, task_labels, time, subjects_to_load, all_data)


% CREATE COLOR SCHEME
% colors = rand(19,3);
colors = [[1 0 0]; [0 1 0]; [0 0 1]; [0.3010 0.7450 0.9330]; [1 0 1]; [1 1 0]; [0 0 0]; [0 0.4470 0.7410]; [0.8500 0.3250 0.0980]; [0.9290 0.6940 0.1250]; [0.4940 0.1840 0.5560]; [0.4660 0.6740 0.1880]; [0 1 1]; [0.6350 0.0780 0.1840]; [0.47 0.25 0.80]; [0.25 0.80 0.54]; [1.00 0.54 0.00]; [0.83 0.14 0.14]; [0.5 0.5 0.5]];

close all;
task_to_select ='Fork_';
ep_to_select ='translation';
% number_synergies = 5;

resampled_scores = cell(size(sorted_scores,1),size(sorted_scores,2));
for it1 = 1:size(sorted_scores,1)
    for it2 = 1:size(sorted_scores,2)
        resampled_scores{it1, it2} = rescale(sorted_scores{it1, it2},-1,1);
    end
end 
   

for subj = 1:numel(subjects_to_load)
   
%    task_idx = strcmp(task_labels{subj},task_to_select);
   task_idx = contains(task_labels{subj},task_to_select);
   ep_idx = strcmp(ep_labels{subj},ep_to_select);

   selection_idx = task_idx & ep_idx;
   
   selected_raw = all_data{subj}(selection_idx, 1:end-3);
   
%    disp([newline newline subjects_to_load{subj}]);
%    disp(['Number of trials with task & EP: ' num2str(sum(selection_idx))]);
   
   if sum(selection_idx) > 0
       % to find if there is more than one EP ocurrence in the task
       aux = find(selection_idx);
       a = aux(1:end-1);
       b = aux(2:end);
       kir = (a-b)+1;
       change = find(kir);

    %    if ~isempty(change)
    %        disp([subjects_to_load(subj), ' at', num2str(change)]);
    %    end
       
       mat = cell2mat(sorted_scores(:,subj)'); % Cast to matrix
%        disp(strcat('Number of NaNs for ', subjects_to_load(subj), ': ', newline, num2str(unique(sum(isnan(mat) ) ) ) ) );
       vari = var(mat, 'omitnan'); % Variance calculation
       [m,i] = sort(vari, 'descend', 'MissingPlacement', 'last'); % Sort by variance
       csum = cumsum(m);
       total_var = max(csum);
       var_09 = total_var * 0.95;
%        var_09 = total_var;
       needed_synergies = find(gt(csum, var_09),1,'first');
       var_expl = csum(needed_synergies) / total_var;
%        disp(['Number of Synergies:' num2str(needed_synergies) ', Variances: ' num2str(var_expl)]);
       syns_to_select = i(1:needed_synergies); % Synergies to select
       new_syn = mat(selection_idx, syns_to_select); % Select synergies
       new_time = time{subj}(selection_idx); % Select timestamps
       
       resampled_mat = cell2mat(resampled_scores(:,subj)'); % Cast to matrix
       resampled_syn = resampled_mat(selection_idx, syns_to_select);
%         resampled_syn = resampled_mat(selection_idx, :);
       
%        disp(['Syns to select: ' num2str(syns_to_select)]);
%        disp(['Size new syn: ' num2str(size(new_syn))]);
       
       if ~isempty(change)
    %        disp([subjects_to_load(subj), ' at', num2str(change)]);
           split = {};
           resampled_split = {};
           raw_split = {};
           s_time = {};
           orig = 1;
           for iter = 1:numel(change)
              split{end+1} = new_syn(orig:change(iter),:);
              resampled_split{end+1} = resampled_syn(orig:change(iter),:);
              raw_split{end+1} = str2double(selected_raw(orig:change(iter),:));
              s_time{end+1} = new_time(orig:change(iter));
              orig = change(iter) + 1;
           end
           split{end+1} = new_syn(orig:end,:);
           resampled_split{end+1} = resampled_syn(orig:end,:);
           raw_split{end+1} = str2double(selected_raw(orig:end,:));
           s_time{end+1} = new_time(orig:end);

       else
           split = {new_syn};
           resampled_split = {resampled_syn};
           raw_split = {str2double(selected_raw)};
           s_time = {new_time};
       end
       
       
       leg = plus("S",string(syns_to_select));
       for x = 1:numel(split)
           
%            init_time = datetime(s_time{x}(1),'ConvertFrom','posixtime','Format','dd-MMM-yyyy HH:mm:ss.SSS');
%            elapsed = [init_time];
%            for iter_time = 2:size(split{x},1)
%                aux_time = datetime(s_time{x}(iter_time),'ConvertFrom','posixtime','Format','dd-MMM-yyyy HH:mm:ss.SSS');
%                elapsed(end+1) = milliseconds(aux_time - init_time);
%            end

%            subject = strrep(subjects_to_load{subj}, '_', ' ');
%            task_name = strrep(task_to_select, '_', ' ');
%            figure;
%            hold on;
%            for xx = 1:size(split{x},2)
% %               plot(elapsed, split{x}(:,xx), 'Color', colors(syns_to_select(xx),:), 'LineWidth', 2);
%                 plot(split{x}(:,xx), 'Color', colors(syns_to_select(xx),:), 'LineWidth', 2);
%            end
% %            disp(['Size "trial" to plot: ' num2str(size(split{x}))]);
%            legend(leg, 'Location', 'best');
%            title([subject ', Task: ' task_name  ', Number of synergies: ' num2str(needed_synergies) newline ' EP: ' ep_to_select ', Episode: ' num2str(x)]);

           subject = strrep(subjects_to_load{subj}, '_', ' ');
           task_name = strrep(task_to_select, '_', ' ');
           info = [subject newline 'Task: ' task_name  ', EP: ' ep_to_select ', Episode: ' num2str(x) newline 'Number of synergies: ' num2str(needed_synergies) ', Variance Explained: ' num2str(round(var_expl,3)*100) '%'];
%            handplot_movement_reconstruction(means(subj,:), sorted_pcs(:, subj, :), resampled_split{x}, syns_to_select, info, raw_split{x});
            if contains(info, 'Subject 9') && x == 3
                handplot_movie(means(subj,:), sorted_pcs(:, subj, :), resampled_split{x}, syns_to_select, info, raw_split{x});
            end
           
       end
   end
end

end