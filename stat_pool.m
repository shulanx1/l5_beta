datapath = 'E:\Code\l5_beta\outputs';
filename = 'input085_percentage03_no_stochastic';

amp = cell(1,4);
dur = cell(1,4);
% num = zeros(6,4);
for i = 1
control_file = load(fullfile(datapath,[sprintf('apicinh%s_%d_stat', filename, i),'.mat']));
amp{1} = control_file.beta_amp;
dur{1} = control_file.beta_dur;
% num(i+1, 1) = control_file.beta_num;
ap5_file = load(fullfile(datapath,[sprintf('noapicalNMDA%s_%d_stat', filename, i),'.mat']));
amp{2} = ap5_file.beta_amp;
dur{2} = ap5_file.beta_dur;
% num(i+1, 2) = ap5_file.beta_num;
noCa_file = load(fullfile(datapath,[sprintf('noapicCa%s_%d_stat', filename, i),'.mat']));
amp{3} = noCa_file.beta_amp;
dur{3} = noCa_file.beta_dur;
% num(i+1, 3) = noCa_file.beta_num;
lowinh_file = load(fullfile(datapath,[sprintf('control%s_%d_stat', filename, i),'.mat']));
amp{4} = lowinh_file.beta_amp;
dur{4} = lowinh_file.beta_dur;
% num(i+1, 4) = lowinh_file.beta_num;
end

for i = 1:4
    amp{i} = amp{i}/1e6;
end
%%
datapath = 'E:\Code\l5_beta\outputs\I_1_E_0_8_apical1_5_basal1_5';
edges = 0:5:200;
freq = [1,3,5,10,15,20,25,30,40,60, 0];
for i = 1:length(freq)
    filename_stat = sprintf('i_mod_%d_stat.mat', freq(i));
    filename = sprintf('i_mod_%d.mat', freq(i));
    load(fullfile(datapath, filename));
    load(fullfile(datapath, filename_stat));
    isi_hist = histcounts(isi, edges);
    % figure, bar(edges(1:end-1) + diff(edges)/2, isi_hist), title(sprintf('%d Hz', freq(i)))
    isi_short(i) = sum(isi_hist(find(edges<20)))/sum(isi_hist);
    isi_long(i) = sum(isi_hist(find((edges>=60) & (edges < 120))))/sum(isi_hist);
    beta_power(i) = sum(abs(hilbert(lfp_beta/1e3)).^2);
    beta_event_count(i) = size(betaBurst, 2);
    beta_amplitude(i) = mean(beta_amp);
    beta_itpc(i) = spike_beta_cohe(40);
    apic_itpc(i) = spike_apicbeta_cohe;
end
freq_plot = [1,3,5,10,15,20,25,30,40,60, 100];
figure
yyaxis left
plot(freq_plot, isi_short)
yyaxis right
plot(freq_plot, beta_itpc)

figure
yyaxis left
plot(freq_plot, beta_power/max(beta_power))
yyaxis right
plot(freq_plot, beta_event_count/20)

