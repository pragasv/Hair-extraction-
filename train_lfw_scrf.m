%%% train spatial conditional random fields on LFW labeling dataset
fprintf('\n\nstart SCRF training\n\n');
fprintf('processing the features!!\n');


Y = cell(length(trainnames),1);
for i = 1:length(trainnames),
    gtfn = sprintf('%s/%s/%s_%04d.dat', gt_dir, trainnames{i}, trainnames{i}, trainnums(i));
    fidgt = fopen(gtfn);
    numNodes = fscanf(fidgt, '%d', 1);
    Y{i} = fscanf(fidgt, '%d', [1 numNodes]) + 1;
    fclose(fidgt);
end
clear X;

for i = 1:length(trainnames),
    if ~mod(i,10), fprintf('.'); end
    if ~mod(i,500), fprintf('%d\n',i); end
    % superpixel features
    % numNodes  : number of superpixels
    % H         : node features
    % E         : adjacent matrix
    % S         : edge features
    [numNodes, H, E, S] = getFeatures(trainnames{i}, trainnums(i), features_dir);
    if rmposfeat,
        H(65:128, :) = [];
    end
    % node features
    H = bsxfun(@rdivide,H,sds);
    H(end+1,:) = 1; % add bias term
    
    % edge features
    [xe, ye] = find(E > 0);
    for j = 1:length(xe),
        S{xe(j),ye(j)} = S{xe(j),ye(j)} ./ esds;
        S{xe(j),ye(j)}(end+1) = 1; % add bias term
    end
    
    X(i) = struct('numNodes', numNodes, 'adjmat', {E}, 'nodeFeatures', {H}, 'edgeFeatures', {S}, 'mapping_block', [], 'mapping_sp',[]);
    
    % read superpixel data
    spfile = sprintf('%s/%s/%s_%04d.dat', spmat_dir, trainnames{i}, trainnames{i}, trainnums(i));
    sp = load(spfile) + 1;
    
    % create projection matrix
    [proj_block, proj_sp] = create_mapping(sp,numNodes,dim_crf,olddim);
    
    X(i).mapping_block = proj_block;
    X(i).mapping_sp = proj_sp';
    clear H E S numNodes proj_block proj_sp;
end


%%% --- Train Spatial CRF with MeanField (edge only) --- %%%
w_scrf.nodeWeights = w_slr.nodeWeights;
params = w_scrf.params;

w_scrf = spatial_crf_train_edge_only(w_scrf, Y, X, dim_crf, nlabel, l2reg_node, l2reg_edge);


%%% --- Train Spatial CRF with MeanField --- %%%
if ~skip_nodeup,
    w_scrf_full = spatial_crf_train(w_scrf, Y, X, dim_crf, nlabel, l2reg_node, l2reg_edge, 1);
else
    w_scrf_full = w_scrf;
end

w_scrf.params = params;
w_scrf_full.params = params;

clear X Y;
