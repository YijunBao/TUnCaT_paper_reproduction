clear
% addpath(genpath('..\..\evaluation'))
ThJaccard = 0.5;
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};

dir_video = '..\data\ABO\';
dir_traces='..\results\ABO\unmixed traces\';
% dir_traces = dir_video;
dir_GTMasks = [dir_video,'GT Masks\'];

list_video = {'Raw','SNR'};
for vid = 1:length(list_video)
    video = list_video{vid};
    root = [dir_traces,'caiman-Batch_',video,'\'];
    if ~exist([root,'Masks\'])
        mkdir([root,'Masks\'])
    end

    FinalRecall = zeros(1,10); FinalPrecision = FinalRecall; FinalF1 = FinalRecall;
    stat_ProcessTime = [];
    for k= 1:10
        load([root,'275\',list_Exp_ID{k},'.mat']); %
        % Save execution time info
        stat_ProcessTime = [stat_ProcessTime;ProcessTime];
        
        % get masks and final performance
        L1=sqrt(size(Ab,1));
        [finalSegments] = ProcessOnACIDMasks(Ab,[L1,L1],0.2);  
        [FinalRecall(k), FinalPrecision(k), FinalF1(k)] = GetPerformance_Jaccard(...
            dir_GTMasks,list_Exp_ID{k},finalSegments,ThJaccard);
        
        %save results
        save([root,'Masks\',list_Exp_ID{k},'_neurons.mat'],'finalSegments','-v7.3')
    end

    ProcessTime = stat_ProcessTime;
    % save results
    save([root,'Performance_275.mat'],...
        'FinalRecall','FinalPrecision','FinalF1','list_Exp_ID','ProcessTime','-v7.3')
    Table_time = ProcessTime;
    save([root,'Table_time.mat'],'Table_time')
    Table = [FinalRecall; FinalPrecision; FinalF1]';
    Table_ext = [Table; mean(Table,1); std(Table,1,1)];
    row=mean(Table,1)
    row=reshape([mean(Table,1); std(Table,1,1)],1,[]);
end