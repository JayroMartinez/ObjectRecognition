function all_data = filter_ep(in_data, ep)

all_data = {};

for iter = 1:numel(in_data)
   
    subj_data = in_data{iter}; 
    
    if ~isempty(ep)
        idx = matches(subj_data(:,end),ep);
        selected_data = subj_data(idx,1:end-1);
    else
        selected_data = subj_data(:,1:end-1);
    end
        
    all_data{end+1} = str2double(selected_data);

end

end