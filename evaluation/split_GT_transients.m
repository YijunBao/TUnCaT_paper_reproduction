clear;
% addpath(genpath('C:\Matlab Files\Unmixing'));
% addpath(genpath('C:\Matlab Files\Filter'));

%% ABO 
list_baseline_std = {'_ksd-psd'}; % '', 
% list_spike_type ={'include'}; % {'include','only','exclude'}; % 
spike_type = 'include'; % {'include','exclude','only'};
sigma_from = 'Raw'; % {'Raw','Unmix'}; 
video='SNR'; % {'Raw','SNR'}
% list_video= {'SNR'};
dir_video='D:\ABO\20 percent 200';
dir_label = 'C:\Matlab Files\TemporalLabelingGUI-master';
dir_label_save = [dir_label,'\split'];
list_Exp_ID={'501484643';'501574836';'501729039';'502608215';'503109347';...
             '510214538';'524691284';'527048992';'531006860';'539670003'};
dFF = h5read('C:\Matlab Files\Filter\GCaMP6f_spike_tempolate_mean.h5','/filter_tempolate')';
dFF = dFF(dFF>exp(-1));
dFF = dFF'/sum(dFF);

num_Exp=length(list_Exp_ID);
folder = sprintf('traces_ours_%s_sigma1_diag11_v1',video);
dir_FISSA = fullfile(dir_video,folder);
useTF = strcmp(video, 'Raw');
if useTF
    kernel=fliplr(dFF);
else
    kernel = 1;
end
thred_ratio=40; % 7;

%% 1p 
list_baseline_std = {'_ksd-psd'}; % '', 
% list_spike_type ={'include'}; % {'include','only','exclude'}; % 
spike_type = '1p'; % {'include','exclude','only'};
sigma_from = 'Raw'; % {'Raw','Unmix'}; 
video='SNR'; % {'Raw','SNR'}
% list_video= {'SNR'};
dir_video='E:\OnePhoton videos\cropped videos\';
dir_label = dir_video;
dir_label_save = [dir_label,'\split'];
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
dFF = h5read('E:\OnePhoton videos\1P_spike_tempolate.h5','/filter_tempolate')';
dFF = dFF(dFF>exp(-1));
dFF = dFF'/sum(dFF);

num_Exp=length(list_Exp_ID);
method = 'ours';
addon = '_merge'; % '_eps=0.1'; % 
folder = sprintf('traces_%s_%s%s',method,video,addon);
dir_FISSA = fullfile(dir_video,folder);
useTF = strcmp(video, 'Raw');
if useTF
    kernel=fliplr(dFF);
else
    kernel = 1;
end
thred_ratio=120; % 7;


%% common
% baseline_std = '_ksd-psd'; % ''; % '_psd'; % 
for bsid = 1:length(list_baseline_std)
    baseline_std = list_baseline_std{bsid};
if contains(baseline_std,'psd')
    std_method = 'psd';  % comp
    if contains(baseline_std,'ksd')
        baseline_method = 'ksd';
    else
        baseline_method = 'median';
    end
else
    std_method = 'quantile-based std comp';
    baseline_method = 'median';
end

% %%
fprintf('Neuro Tools');
for ind = 1:num_Exp
    Exp_ID = list_Exp_ID{ind};
    fprintf('\b\b\b\b\b\b\b\b\b\b\b%s: ',Exp_ID);
    load(fullfile(dir_label,['output_',Exp_ID,'.mat']),'output');
    if strcmp(spike_type,'exclude')
        for oo = 1:length(output)
            if ~isempty(output{oo}) && all(output{oo}(:,3))
                output{oo}=[];
            end
        end
    elseif strcmp(spike_type,'only')
        for oo = 1:length(output)
            if ~isempty(output{oo}) && ~all(output{oo}(:,3))
                output{oo}=[];
            end
        end
    end

    load(fullfile(dir_FISSA,'raw',[Exp_ID,'.mat']),'traces', 'bgtraces');
%             bgtraces = 0*bgtraces;
    if useTF
        traces_raw_filt=conv2(traces'-bgtraces',kernel,'valid');
    else
        traces_raw_filt=traces'-bgtraces';
    end
    [mu, sigma] = SNR_normalization(traces_raw_filt,std_method,baseline_method);

%     %%
    traces_eval = traces_raw_filt;
    T=size(traces_eval,2);
    % ncells=length(output);
    warning off

%     %% Get GT spikes from manual output.mat
%     hasGT=~cellfun(@isempty, output);
%     num_GT=sum(hasGT);
%     ind_hasGT=find(hasGT);
    num_GT = length(output);
    output_split = output;
    spikes_GT_array=cell(num_GT,1);
%     traces_eval=traces_eval(hasGT,:);
%     mu=mu(hasGT);
%     sigma=sigma(hasGT);
    thred=mu+sigma*thred_ratio;
    spikes_eval_line=traces_eval>thred;
    % spikes_eval_line=spikes_eval_line(hasGT,:);
    spikes_eval_array=cell(num_GT,1);

    for nn=1:num_GT
        if ~isempty(output{nn})
%             %%
            spikes_line=spikes_eval_line(nn,:);
            spikes_line_diff=diff([0,spikes_line,0]);
            starts=find(spikes_line_diff==1);
            ends=find(spikes_line_diff==-1)-1;
            num_active = length(starts);
        %     ends(ends>T)=T;
        %     temp_spikes_eval_array=[starts;ends;ones(1,num_active)]';

            trace=traces_eval(nn,:);
        %     [peaks, locs] = findpeaks(trace, 1:T, 'MinPeakDistance', 5, 'MinPeakProminence', sigma(nn)*3, 'MinPeakHeight', mu(nn)+sigma(nn)*3); %
            [~, locs] = findpeaks(trace, 1:T, 'MinPeakProminence', sigma(nn)*thred_ratio/3, 'MinPeakHeight', thred(nn)); %
            locs = [1,locs,T];
            spikes_peaks=false(1,T);
            spikes_peaks(locs)=true;
            list_temp_spikes_split = cell(num_active,1);
            for ii=1:num_active
                num_spikes = sum(spikes_peaks(starts(ii):ends(ii)));
                if num_spikes==0
        %             temp_spikes_eval_array(ii,3)=0;
                    temp_spikes_split = zeros(0,3);
                elseif num_spikes==1
        %             temp_spikes_eval_array(ii,3)=0;
                    temp_spikes_split = [starts(ii),ends(ii),0];
                elseif num_spikes>1
        %             multi_spike = trace(starts(ii):ends(ii));
                    list_locs = locs((locs >= starts(ii)) & (locs <= ends(ii)));
                    temp_spikes_split = zeros(num_spikes,3);
                    temp_spikes_split(1,1) = starts(ii);
                    temp_spikes_split(num_spikes,2) = ends(ii);
                    for jj = 1:num_spikes-1
                        [~,pos] = min(trace(list_locs(jj):list_locs(jj+1)));
                        valley = pos+list_locs(jj)-1;
                        temp_spikes_split(jj,2) = valley-1;
                        temp_spikes_split(jj+1,1) = valley+1;
                    end
                end
                list_temp_spikes_split{ii} = temp_spikes_split;
            end
            spikes_eval_array{nn} = cell2mat(list_temp_spikes_split);
        %     spikes_eval_array{nn}=temp_spikes_eval_array(logical(temp_spikes_eval_array(:,3)),:);
        %     spikes_eval_array{nn}(:,3)=0;

%             %%
            output_temp=output{nn};
            output_temp(:,2) = output_temp(:,2) - 1;
            num_active = size(output_temp,1);
%             spikes_GT_array{nn}=[output_temp,zeros(num_active,1)];

            list_temp_spikes_split = cell(num_active,1);
            for ii=1:num_active
                num_spikes = sum(spikes_peaks(output_temp(ii,1):output_temp(ii,2)));
                if num_spikes==0
        %             temp_spikes_eval_array(ii,3)=0;
                    temp_spikes_split = zeros(0,3);
                elseif num_spikes==1
        %             temp_spikes_eval_array(ii,3)=0;
                    temp_spikes_split = output_temp(ii,:);
                elseif num_spikes>1
        %             multi_spike = trace(starts(ii):ends(ii));
                    list_locs = locs((locs >= output_temp(ii,1)) & (locs <= output_temp(ii,2)));
                    temp_spikes_split = ones(num_spikes,3)*output_temp(ii,3);
                    temp_spikes_split(1,1) = output_temp(ii,1);
                    temp_spikes_split(num_spikes,2) = output_temp(ii,2);
                    for jj = 1:num_spikes-1
                        [~,pos] = min(trace(list_locs(jj):list_locs(jj+1)));
                        valley = pos+list_locs(jj)-1;
                        temp_spikes_split(jj,2) = valley-1;
                        temp_spikes_split(jj+1,1) = valley+1;
                    end
                end
                list_temp_spikes_split{ii} = temp_spikes_split;
            end
            output_split{nn} = cell2mat(list_temp_spikes_split);
        end
    end
    output=output_split;
    save(fullfile(dir_label_save,['output_',Exp_ID,'.mat']),'output');
end
end
fprintf('\b\b\b\b\b\b\b\b\b\b\b');

