function boxplot_with_datapoint(y, colors)
% y: 2D matrix (each dataset in each column) or cells of data
% colors: color to be used, N*3 matrix, each color in each row

% example
% y = rand(100,2); % matrix form of y
% y = {rand(100,1), rand(150,1)}; % cell form of y
% boxplot_with_datapoint(y);


if nargin < 2
    colors = [[0,0,0];[119,177,204];[61,139,191];[6,50,99]];
    colors = colors/256;
end

if ~iscell(y)
    data = cell(size(y, 2),1);
    for i = 1:size(y,2)
        data{i} = y(:,i);
    end
else
    data = cell(length(y),1);
    for i = 1:length(y)
        if size(y{i},2)==1
            data{i} = y{i};
        else
            data{i} = y{i}';
        end
    end
end

grp = [];
for i = 1:length(data)
    grp = [grp, i*ones(1, length(data{i}))];
end


size_x = (0.4 + length(data)*0.15)/2; % figure size in inches
size_y = 1; 
f = figure();
f.Position(3) = f.Position(4)*size_x;
f.Renderer = 'painters';

for i = 1:length(data)
    color_idx = mod(i, size(colors, 1));
    if color_idx == 0
        color_idx = size(colors, 1);
    end
    data_temp = data{i};
    data_temp(isoutlier(data_temp,'quartiles')) = [];
    scatter(i+(rand(1, length(data_temp))-0.5)/5,data_temp,20,'MarkerFaceColor', colors(color_idx,:),'MarkerEdgeColor', 'None')
    hold on
end
% boxplot(cell2mat(data), grp, 'Color', 'k');
boxplot(cell2mat(data), grp, 'Color', 'k','symbol', '')

end