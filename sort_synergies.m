function [sorted_syn,sorted_var] = sort_synergies(synergies, explained_variance)

% SORT_SYNERGIES Function to load data corresponding to a single subject.
%
% INPUT
%
% synergies:            Array with the synergies. Each value in position 
%                       (i,j) represents the PC for subject 'j' that 
%                       belongs to synergy 'i'.
%
% explained_variance:   Array with the variation for each PC. Each value in
%                       position (i,j) represents the variance explained 
%                       PC 'i' in subject 'j'.
%
% OUTPUT
% sorted_syn:           Synergies ordered regarding the variation that they
%                       explain. Each row represents a synergy and each
%                       column represents a subject. Values in position
%                       (i,j) are the PC from subject 'j' that belongs to
%                       synergy 'i'. Synergies are sorted in descending
%                       order by its mean explained variance.
%
% sorted_var:           Variance accounted for each synergy. It should be
%                       noticed that this variance is calculated as 
%                       the mean between the variances of the PCs which 
%                       form it. 
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          22/07/21
% LAST MODIFIED:    22/07/21

var = [];

for iter = 1:size(synergies,1)
    
   aux_var = [];
   
   for subj = 1:size(synergies,2)
      
       if ~isnan(synergies(iter,subj))
          aux_var = [aux_var, explained_variance(synergies(iter,subj),subj)]; 
       end
       
   end
   
   var = [var; mean(aux_var)];
    
end

[sorted_var, order] = sort(var, 'descend');
sorted_syn = synergies(order,:);
end