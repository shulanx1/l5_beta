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