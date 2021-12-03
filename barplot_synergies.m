function barplot_synergies(synergies, joint_names, subjects_to_load, coeffs)

% BARPLOT_SYNERGIES Creates a barplot for each synergy containing all subjects
%                present in that synergy.
%
% INPUT
% synergies:        NxM matrix where each row represents a synergy and each
%                   column represents a subject. Each position (i,j)
%                   represents the PC for subject 'j'that belongs to
%                   synergy 'i'.
%
% joint_names:      A cell array containing the joint names.
%
% subjects_to_load: A cell array containing the subjects names.
%
% coeffs:           Cell array with a position for each subject. 
%                   Each position contains the coefficients for that
%                   particular subject as  a NxM matrix with each row
%                   representing a joint and each column representing a PC.
%                   Each position (i,j) in this matrix is the coefficient
%                   for joint 'i' that belongs to PC 'j'.
% 
%
% AUTHOR:           Jayro Martinez-Cervero
% CREATED:          22/07/21
% LAST MODIFIED:    22/07/21


% Clean joint names
joint_names = regexprep(joint_names, '\w_', '');

for syn = 1:size(synergies,1)
    
    subject_in_syn = [];
    coeff_to_plot = [];
   
    for subj = 1:size(synergies,2)
    
        if ~isnan(synergies(syn,subj))
            subject_in_syn = [subject_in_syn; {subjects_to_load{subj}}];
            subj_coeffs = coeffs{subj};
            coeff_to_plot = [coeff_to_plot,subj_coeffs(:,synergies(syn,subj))];
            a=1;
        end
        
    end
    
    figure;
    b = bar(coeff_to_plot);
    set(b, {'DisplayName'}, subject_in_syn);
    set(gca,'xtick',1:numel(joint_names));
    set(gca,'XTickLabel',joint_names);
    xtickangle(45);
    title(['Synergy ' num2str(syn)]);
    legend('Location', 'best', 'Interpreter', 'none');

end
% 
% % S1
% figure;
% b = bar([s3_coeff(:,1),s4_coeff(:,1)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 1');
% legend('Location', 'best');
% 
% 
% % S2
% figure;
% subplot(2,2,1);
% b = bar([s3_coeff(:,2),s4_coeff(:,3)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 2');
% % legend('Location', 'best');
% 
% % S3
% % figure;
% subplot(2,2,2);
% b = bar([s3_coeff(:,3),s4_coeff(:,2)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 3');
% % legend('Location', 'best');
% 
% % S4
% % figure;
% subplot(2,2,3);
% b = bar([s3_coeff(:,4),s4_coeff(:,7)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 4');
% % legend('Location', 'best');
% 
% % S5
% % figure;
% subplot(2,2,4);
% b = bar([s3_coeff(:,5),s4_coeff(:,5)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 5');
% legend('Location', 'best');
% 
% % S6
% figure;
% subplot(2,2,1);
% b = bar([s3_coeff(:,6),s4_coeff(:,6)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 6');
% % legend('Location', 'best');
% 
% % S7
% % figure;
% subplot(2,2,2);
% b = bar([s3_coeff(:,7),s4_coeff(:,10)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 7');
% % legend('Location', 'best');
% 
% % S8
% % figure;
% subplot(2,2,3);
% b = bar([s3_coeff(:,8),s4_coeff(:,13)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 8');
% % legend('Location', 'best');
% 
% % S9
% % figure;
% subplot(2,2,4);
% b = bar([s3_coeff(:,9),s4_coeff(:,12)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 9');
% legend('Location', 'best');
% 
% % S10
% figure;
% subplot(2,2,1);
% b = bar([s3_coeff(:,10),s4_coeff(:,9)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 10');
% % legend('Location', 'best');
% 
% % S11
% % figure;
% subplot(2,2,2);
% b = bar([s3_coeff(:,11),s4_coeff(:,8)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 11');
% % legend('Location', 'best');
% 
% % S12
% % figure;
% subplot(2,2,3);
% b = bar([s3_coeff(:,12),s4_coeff(:,14)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 12');
% % legend('Location', 'best');
% 
% % S13
% % figure;
% subplot(2,2,4);
% b = bar([s3_coeff(:,13),s4_coeff(:,15)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 13');
% legend('Location', 'best');
% 
% % S14
% figure;
% subplot(2,2,1);
% b = bar([s3_coeff(:,14),s4_coeff(:,4)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 14');
% % legend('Location', 'best');
% 
% % S15
% % figure;
% subplot(2,2,2);
% b = bar([s3_coeff(:,15),s4_coeff(:,16)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 15');
% % legend('Location', 'best');
% 
% % S16
% % figure;
% subplot(2,2,3);
% b = bar([s3_coeff(:,16),s4_coeff(:,11)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 16');
% % legend('Location', 'best');
% 
% % S17
% % figure;
% subplot(2,2,4);
% b = bar([s3_coeff(:,17),s4_coeff(:,18)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 17');
% legend('Location', 'best');
% 
% % S18
% figure;
% b = bar([s3_coeff(:,18),s4_coeff(:,17)]);
% b(1).FaceColor = 'red';
% b(2).FaceColor = 'blue';
% set(b, {'DisplayName'}, {'Subject 3'; 'Subject 4'});
% set(gca,'xtick',1:numel(joint_names));
% set(gca,'XTickLabel',joint_names);
% xtickangle(45);
% title('Synergy 18');
% legend('Location', 'best');

end