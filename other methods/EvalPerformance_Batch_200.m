clear
% Path to where "GetPerformance_Jaccard.m" is located
addpath(genpath(''))
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))
% Run over Layer 275 data
ThJaccard = 0.5;
expID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
% gtDir = 'C:\Matlab Files\STNeuroNet-master\Markings\ABO\Layer275\FinalGT\';
gtDir = 'D:\ABO\20 percent 200\GT Masks\';

root = 'D:\ABO\20 percent 200\caiman-Batch_SNR\'; % bin 5
if ~exist([root,'Masks\'])
    mkdir([root,'Masks\'])
end

FinalRecall = zeros(1,10); FinalPrecision = FinalRecall; FinalF1 = FinalRecall;
stat_ProcessTime = [];
for k= 1:10
    load([root,'275\',expID{k},'.mat']); %
    % Save execution time info
    stat_ProcessTime = [stat_ProcessTime;ProcessTime];
    
    % get masks and final performance
    L1=sqrt(size(Ab,1));
    [finalSegments] = ProcessOnACIDMasks(Ab,[L1,L1],0.2);  
    [FinalRecall(k), FinalPrecision(k), FinalF1(k)] = GetPerformance_Jaccard(...
        gtDir,expID{k},finalSegments,ThJaccard);
    
    %save results
    save([root,'Masks\',expID{k},'_neurons.mat'],'finalSegments','-v7.3')
end

ProcessTime = stat_ProcessTime;
% save results
% save([root,'Performance_275.mat'],...
%     'FinalRecall','FinalPrecision','FinalF1','expID','ProcessTime','-v7.3')
Table_time = ProcessTime;
save([root,'Table_time.mat'],'Table_time')
Table = [FinalRecall; FinalPrecision; FinalF1]';
Table_ext = [Table; mean(Table,1); std(Table,1,1)];
row=mean(Table,1)
row=reshape([mean(Table,1); std(Table,1,1)],1,[]);
