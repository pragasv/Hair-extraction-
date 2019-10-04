%% 
% generate_sds.m
% 
% This script is used to compute standard deviations for scaling node/edge features.
%
    
%%

startup;

%% Load training data

fprintf('Loading Data...\n');

for i=1:num_train
    if (mod(i,100) == 0)
        fprintf('%d\n', i);
    end
    
    [numNodes H E S] = getFeatures(trainnames{i}, trainnums(i), features_dir);
    X(i) = struct('numNodes', numNodes, ...
        'adjmat', {E}, ...
        'nodeFeatures', {H}, ...
        'edgeFeatures', {S});
end

%Node features is num_node_features x number of nodes (superpixels)
numNodeFeatures = size(X(1).nodeFeatures, 1);

%find the edges
[xe,ye] = find(X(1).adjmat > 0);

numEdgeFeatures = [];

if (isempty(xe) && isempty(ye))    
    numEdgeFeatures = 0;
else
    numEdgeFeatures = length(X(1).edgeFeatures{xe(1),ye(1)});
end

%% Compute standard deviations

fprintf('Computing standard deviations...\n');

%total number of superpixels
tot = 0;

sds = zeros(numNodeFeatures,1);

%compute the sum "across" the rows, for all superpixels for each node feature
for i=1:num_train
  sds = sds + sum(X(i).nodeFeatures.^2,2);
  tot = tot + size(X(i).nodeFeatures,2);
end

sds = sds / tot;
sds = sqrt(sds);
sds(sds < .0000001) = 1;

%Compute same for edges
tot = 0;
esds = zeros(numEdgeFeatures,1);
for i=1:num_train
  [xe, ye] = find(X(i).adjmat > 0);
  for j=1:length(xe)
    esds = esds + X(i).edgeFeatures{xe(j),ye(j)}.^2;
  end
  tot = tot + length(xe);
end
esds = esds / tot;
esds = sqrt(esds);
esds(esds < .0000001) = 1;

%% Save features

save('common/sds_large.mat', 'sds');
save('common/esds_large.mat', 'esds');