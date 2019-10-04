%%% train conditional RBM on LFW labeling dataset
fprintf('\n\nstart GLOC training\n\n');
fprintf('processing the features!!\n');


numDim = size(w_scrf.nodeWeights,1);
params = gloc_params(dataname,fname_gloc,numDim,nlabel,dim_crf,l2reg_node,l2reg_edge,dim_rbm,numHid,l2reg,epsilon_gloc,KCD_gloc,anneal_gloc,0,batchSize,maxepoch_gloc,fsave_dir);

% load label, superpixels, features
tr_label = cell(length(trainnames),1);
tr_feat = cell(length(trainnames),1);
tr_proj_rbm = cell(length(trainnames),1);
tr_proj_crf = cell(length(trainnames),1);

for i = 1:length(trainnames),
    if ~mod(i,10), fprintf('.'); end
    if ~mod(i,500), fprintf('%d\n',i); end
    
    % load full data
    gt_casename = sprintf('%s/%s/%s_%04d.dat', gt_dir, trainnames{i}, trainnames{i}, trainnums(i));
    gt_case = load(gt_casename);
    gt_case = gt_case + 1;         % increase offset by 1
    gt_splabels = gt_case(2:end);   % the first value is the number of nodes
    tr_label{i} = single(gt_splabels);
    
    % read superpixel features
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
    X = struct('numNodes', numNodes, 'adjmat', {E}, 'nodeFeatures', {H}, 'edgeFeatures', {S});
    [~, num_sp] = size(X.nodeFeatures);
    
    tr_feat{i} = X;
    
    % read superpixel data
    spfile = sprintf('%s/%s/%s_%04d.dat', spmat_dir, trainnames{i}, trainnames{i}, trainnums(i));
    sp = load(spfile) + 1;
    
    % create projection matrix
    [proj_blk, ~] = create_mapping(sp,num_sp,dim_rbm,olddim);
    tr_proj_rbm{i} = single(proj_blk);
    
    [~, proj_sp] = create_mapping(sp,num_sp,dim_crf,olddim);
    tr_proj_crf{i} = single(proj_sp);
    
    clear gt_splabels sp X;
end


[w_gloc, w_rbm, w_scrf, params] = gloc_train_percLoss(tr_feat, tr_proj_crf, tr_proj_rbm, tr_label, params, w_scrf, w_rbm);
w_gloc.params = params;
w_gloc.params.rmposfeat = rmposfeat;

clear tr_feat tr_proj_crf tr_proj_rbm tr_label;
