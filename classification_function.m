function classification_function(timepoints, ep_labels, task_labels)

%% INITIAL ASSIGNMENTS
clc;

families = {'Mugs'; 'Plates'; 'Geometric'; 'Cutlery'; 'Ball'};
objects = [{'CeramicMug', 'Glass', 'MetalMug'}; {'CeramicPlate','MetalPlate','PlasticPlate'}; {'Cube','Cylinder','Triangle'}; {'Fork','Knife','Spoon'}; {'PingPongBall','SquashBall','TennisBall'}];
eps = ["contour following"; "contour following + enclosure part"; "edge following"; "enclosure"; "enclosure part"; "enclosure part + function test"; "function test"; "pressure"; "rotation"; "translation"; "weighting"; "weighting + contour following"];

family_to_select = 'Mugs';

num_bins = 5;
cv = 10;

%% TO TRIALS
all_timepoints = cat(1,timepoints{:});
all_tasks = cat(1,task_labels{:});
all_eps = cat(1,ep_labels{:});
categorical_tasks = grp2idx(categorical(all_tasks));
changes = find(diff(categorical_tasks));

init = 1;
trials = cell(1,numel(changes)+1);
task_per_trial = [];
eps_per_trial = cell(1,numel(changes)+1);

for iter = 1:numel(changes)
    trials{iter} = all_timepoints(init:changes(iter),:);
    task_per_trial = [task_per_trial; all_tasks(init)];
    eps_per_trial{iter} = all_eps(init:changes(iter),:);
    init = changes(iter)+1;
end

trials{end} = all_timepoints(init:end,:);
task_per_trial = [task_per_trial; all_tasks(init)];
eps_per_trial{end} = all_eps(init:end,:);

sp_obj = split(task_per_trial, '_');
given_object = sp_obj(:,1);
target_object = sp_obj(:,2);

%% SELECT BY FAMILY
trials_to_select = contains(given_object, objects(contains(families, family_to_select),:));
filtered_trials = trials(trials_to_select);
filtered_eps = eps_per_trial(trials_to_select);
filtered_given = given_object(trials_to_select);

%% DIVIDE TRIALS BY EPs & BINS
max_eps_trial = 0;

splitted_trials = cell(1,numel(filtered_trials));
splitted_eps = cell(1,numel(filtered_trials));

for tr = 1:numel(filtered_trials)
    
    tr_eps = filtered_eps{tr};
    categorical_eps = grp2idx(categorical(tr_eps));
    chang = find(diff(categorical_eps));
    
    if numel(chang)+1 > max_eps_trial
        max_eps_trial = numel(chang)+1;
    end
    
    in = 1;
    s_trial = filtered_trials{tr};
    spl_trial = cell(1,numel(chang)+1);
    ep_perf = [];
    
    for ch = 1:numel(chang)+1
        
        ep_perf = [ep_perf; tr_eps(in)];
        
        if ch <= numel(chang)
            each_ep = s_trial(in:chang(ch),:);
        else
            each_ep = s_trial(in:end,:);
        end
        
        bins_bound = ceil(linspace(1,size(each_ep,1),num_bins));
        bins = [];
        
        for b = 1:num_bins-1
           bins = [bins; mean(each_ep(bins_bound(b):bins_bound(b+1),:))];
        end
        
        bins = [bins; mean(each_ep(bins_bound(b):bins_bound(end),:))];
        spl_trial{ch} = bins;
        
        if ch <= numel(chang)
            in = chang(ch) + 1;
        end
    end
    
    splitted_trials{tr} = spl_trial;
    splitted_eps{tr} = ep_perf';
    
end

%% CLASSIFICATION
part = cvpartition(filtered_given,'KFold',cv);
acc = [];

for kf = 1:part.NumTestSets
    
    tr_idx = training(part, kf);
    ts_idx = test(part, kf);

    train_trials = splitted_trials(tr_idx);
    train_obj = filtered_given(tr_idx);

    test_trials = splitted_trials(ts_idx);
    test_obj = filtered_given(ts_idx);
    
    train_dat = [];
    train_lab = [];
    
    for i=1:numel(train_trials)
        
        for j=1:numel(train_trials{i})
           
            tmp_dat = reshape(train_trials{i}{j},1,[]);
            train_dat = [train_dat; tmp_dat];
            train_lab = [train_lab; train_obj(i)];
            
        end
        
    end
    
    test_dat = [];
    test_lab = [];
    
    for i=1:numel(test_trials)
        
        for j=1:numel(test_trials{i})
           
            tmp_ts_dat = reshape(test_trials{i}{j},1,[]);
            test_dat = [test_dat; tmp_ts_dat];
            test_lab = [test_lab; test_obj(i)];
            
        end
        
    end
    
    
%     [B,dev,stats] = mnrfit(train_dat, categorical(train_lab), 'model', 'nominal');
%     predi = mnrval(B,test_dat);

%     learners = ['discriminant','kernel','knn','linear','naivebayes','svm','tree'];
    method = 'svm';
    model = fitcecoc(train_dat,categorical(train_lab),'Learners',method);
    error = resubLoss(model);
    disp(['ERROR: ' num2str(error)]);
    predi = predict(model,test_dat);
    results = predi == categorical(test_lab);
    acc = [acc round((sum(results)/numel(results))*100,2)];
    
    
end
disp([newline 'FAMILY: ' family_to_select ' METHOD: ' method ' CV:  ' num2str(cv) ' Bins: ' num2str(num_bins)]);
disp(['Accuracy: ' num2str(acc) ' %']);
disp(['Mean Accuracy: ' num2str(mean(acc)) ' %']);

end