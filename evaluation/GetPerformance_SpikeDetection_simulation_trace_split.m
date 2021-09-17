function [recall, precision, F1,individual_recall,individual_precision,...
    spikes_GT_array,spikes_eval_array] ...
    = GetPerformance_SpikeDetection_simulation_trace_split(spikes_GT_array,traces_eval,thred_ratio,sigma,mu)
% Calculate recall, precision, and F1 between manually labelled spikes and program generated spikes from traces
% calcium is the clean trace.
% thred_ratio is the threshold ratio for spike detection.
% traces_eval is the trace to be evaluated (spikes are generated from it).
% traces_raw is the trace used to claculate sigma. When traces_eval is generated through NMF unmixing,
    % it is better to use the original trace to calculate sigma. 
    % If traces_raw is empty, then traces_eval will be used.

% if nargin<4
%     traces_raw=traces_eval;
% end
T=size(traces_eval,2);
% T=23090;
% ncells=length(output);
warning off
hasGT=~cellfun(@isempty, spikes_GT_array);
num_GT=sum(hasGT);
spikes_GT_array=spikes_GT_array(hasGT);

%% Generate GT transients from clean traces
% calcium=calcium(hasGT,:);
% [mu_clean, sigma_clean] = SNR_normalization(calcium,'std','median');
% peaks_avg = mu_clean;
% for nn = 1:num_GT
%     trace_clean = calcium(nn,:);
%     [peaks,locs] = findpeaks(trace_clean,'MinPeakProminence',sigma_clean(nn));
%     [f,xi] = ksdensity(peaks);
%     maxf = max(f); 
%     pos = find(f==maxf);
%     peaks_avg(nn) = mean(xi(pos)) - mu_clean(nn);
% end
% peaks_amp = mean(peaks_avg);
% thred=mu_clean+peaks_amp/2;
% spikes_GT_line=calcium>thred;
% spikes_GT_array=cell(num_GT,1);
% for nn=1:num_GT
%     spikes_line=spikes_GT_line(nn,:);
%     spikes_line_diff=diff([0,spikes_line,0]);
%     starts=find(spikes_line_diff==1);
%     ends=find(spikes_line_diff==-1)-1;
% %     ends(ends>T)=T;
%     spikes_GT_array{nn}=[starts;ends;zeros(size(starts))]';
% %     temp_spikes_GT_array=[starts;ends;ones(size(starts))]';
% %     
% %     trace=calcium(nn,:);
% % %     [peaks, locs] = findpeaks(trace, 1:T, 'MinPeakDistance', 5, 'MinPeakProminence', sigma(nn)*3, 'MinPeakHeight', mu(nn)+sigma(nn)*3); %
% %     [peaks, locs] = findpeaks(trace, 1:T, 'MinPeakProminence', peaks_amp/3, 'MinPeakHeight', thred(nn)); %
% %     spikes_peaks=false(1,T);
% %     spikes_peaks([1,locs,end])=true;
% %     for ii=1:length(starts)
% %         if ~sum(spikes_peaks(starts(ii):ends(ii)))
% %             temp_spikes_GT_array(ii,3)=0;
% %         end
% %     end
% %     spikes_GT_array{nn}=temp_spikes_GT_array(logical(temp_spikes_GT_array(:,3)),:);
% %     spikes_GT_array{nn}(:,3)=0;
% end

% %%
% hasGT=~cellfun(@isempty, output);
% num_GT=sum(hasGT);
% ind_hasGT=find(hasGT);
% spikes_GT_array=cell(num_GT,1);
% spikes_GT_line=false(num_GT,T);
% for nn=1:num_GT
%     if ~isempty(output{ind_hasGT(nn)})
%         output_temp=output{ind_hasGT(nn)};
%         output_temp=output_temp(logical(output_temp(:,3)),1:2);
%         output_temp(:,2) = output_temp(:,2) - 1;
%         spikes_GT_array{nn}=[output_temp,zeros(size(output_temp,1),1)];
%         for ii=1:size(output_temp,1)
%             spikes_GT_line(nn,output_temp(ii,1):output_temp(ii,2))=true;
%         end
%     end
% end

%% Get detected spikes
traces_eval=traces_eval(hasGT,:);
mu=mu(hasGT);
sigma=sigma(hasGT);
thred=mu+sigma*thred_ratio;
spikes_eval_line=traces_eval>thred;
% spikes_eval_line=spikes_eval_line(hasGT,:);
spikes_eval_array=cell(num_GT,1);
for nn=1:num_GT
    spikes_line=spikes_eval_line(nn,:);
    spikes_line_diff=diff([0,spikes_line,0]);
    starts=find(spikes_line_diff==1);
    ends=find(spikes_line_diff==-1)-1;
    num_active = length(starts);
%     ends(ends>T)=T;
%     temp_spikes_eval_array=[starts;ends;ones(size(starts))]';
    
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
                [~,ind] = min(trace(list_locs(jj):list_locs(jj+1)));
                valley = ind+list_locs(jj)-1;
                temp_spikes_split(jj,2) = valley-1;
                temp_spikes_split(jj+1,1) = valley+1;
            end
        end
        list_temp_spikes_split{ii} = temp_spikes_split;
    end
    spikes_eval_array{nn} = cell2mat(list_temp_spikes_split);
%     spikes_eval_array{nn}=temp_spikes_eval_array(logical(temp_spikes_eval_array(:,3)),:);
%     spikes_eval_array{nn}(:,3)=0;
end

%% Compare GT and detected spikes
n_GT_total=cellfun(@(x) size(x,1), spikes_GT_array);
n_eval_total=cellfun(@(x) size(x,1), spikes_eval_array);
n_GT_match=zeros(num_GT,1);
n_eval_match=zeros(num_GT,1);
for nn=1:num_GT
%     spikes_GT1=spikes_GT_line(nn,:);
    spikes_GT2=spikes_GT_array{nn};
    n_GT = size(spikes_GT2,1);
%     spikes_eval1=spikes_eval_line(nn,:);
    spikes_eval2=spikes_eval_array{nn};
    n_eval = size(spikes_eval2,1);
    overlap = zeros(n_GT,n_eval);
    for ii = 1:n_GT
        spike_GT_ii = spikes_GT2(ii,:);
        for jj = 1:n_eval
            spike_eval_ii = spikes_eval2(jj,:);
            overlap(ii,jj) = max(0,1+min(spike_GT_ii(2),spike_eval_ii(2))-max(spike_GT_ii(1),spike_eval_ii(1)));
        end
    end
    dist = -overlap;
    dist(overlap==0)=inf;
    [m,~] = Hungarian(dist);
%     m(overlap==0)=0;
    
%     GT_match=zeros(size(spikes_GT2,1),1);
%     for ii=1:size(spikes_GT2,1)
%         GT_match(ii)=sum(spikes_eval_line(nn,spikes_GT2(ii,1):spikes_GT2(ii,2)))>0;
%     end
    GT_match = any(m,2);
    n_GT_match(nn)=sum(GT_match);
    spikes_GT_array{nn}(:,3)=GT_match;
    
%     eval_match=zeros(size(spikes_eval2,1),1);
%     for ii=1:size(spikes_eval2,1)
%         eval_match(ii)=sum(spikes_GT_line(nn,spikes_eval2(ii,1):spikes_eval2(ii,2)))>0;
%     end
    eval_match = any(m,1)';
    n_eval_match(nn)=sum(eval_match);
    spikes_eval_array{nn}(:,3)=eval_match;
end
    
individual_recall=n_GT_match./n_GT_total;
individual_precision=n_eval_match./n_eval_total;
recall=sum(n_GT_match)/sum(n_GT_total);
precision=sum(n_eval_match)/sum(n_eval_total);
F1=2/(1/recall+1/precision);
F1(isnan(F1))=0;


