function evo_weight_task(sorted_scores, sorted_pcs, ep_labels, task_labels, subjects_to_load)

close all;

task_to_select = 'PingPongBall_TennisBall';
% task_to_select ='CeramicMug_CeramicMug';
% task_to_select ='Cube_Cylinder';

number_synergies = 5;

for subj = 1:numel(subjects_to_load) 
   
    task_idx = strcmp(task_labels{subj},task_to_select);
    
    eps_by_task = ep_labels{subj}(task_idx);
    aux_eps = [eps_by_task(1); eps_by_task(1:end-1)];
    difer = find(ne(eps_by_task,aux_eps))';
    
    new_syn =[];
    for syn = 1:number_synergies
        new_syn = [new_syn sorted_scores{syn,subj}(task_idx)];
    end
    
    subject = strrep(subjects_to_load{subj}, '_', ' ');
    task_name = strrep(task_to_select, '_', ' ');
    figure;
    hold on;
    plot(new_syn, 'LineWidth', 2);
    xline(0,'--', char(eps_by_task(1)), 'LabelHorizontalAlignment', 'right', 'FontSize', 12);
%     arrayfun(@(a) xline(a,'-.', char(eps_by_task(a-1)), 'LabelHorizontalAlignment', 'left'), difer);
    arrayfun(@(a) xline(a,'-.', char(eps_by_task(a+1)), 'LabelHorizontalAlignment', 'right', 'FontSize', 12), difer);
    legend('S1','S2','S3','S4','S5', 'Location', 'best');
    title([subject ', Task: ' task_name ', Number of synergies: ' num2str(number_synergies)]);
    
end

end