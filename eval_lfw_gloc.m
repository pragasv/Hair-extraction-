function [acc, evaltime] = eval_lfw_gloc(w_gloc, datanames, datanums, sds, esds, verbose)
startup_directory;
olddim = 250;
addpath('model/crbm');

w_gloc.vishidrs = reshape(w_gloc.vishid,size(w_gloc.vishid,1)*size(w_gloc.vishid,2),size(w_gloc.vishid,3));
w_gloc.visbiasrs  = reshape(w_gloc.visbiases,size(w_gloc.vishid,1)*size(w_gloc.vishid,2),1);
w_gloc.nodeWeightsrs = reshape(w_gloc.nodeWeights,size(w_gloc.nodeWeights,1)*size(w_gloc.nodeWeights,2),size(w_gloc.nodeWeights,3));
w_gloc.vishidperm = reshape(w_gloc.vishid,size(w_gloc.vishid,1),size(w_gloc.vishid,2)*size(w_gloc.vishid,3));

tot_err = 0;
tot_sp = 0;
evaltime = 0;
for i = 1:length(datanums),
    % load full data
    gt_casename = sprintf('%s/%s/%s_%04d.dat', gt_dir, datanames{i}, datanames{i}, datanums(i));
    gt_case = load(gt_casename);
    gt_case = gt_case + 1;         % increase offset by 1
    gt_splabels = gt_case(2:end);   % the first value is the number of nodes
    
    % read superpixel features
    [numNodes, H, E, S] = getFeatures(datanames{i}, datanums(i), features_dir);
    if w_gloc.params.rmposfeat,
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
    num_sp = numNodes;
    
    % read superpixel data
    spfile = sprintf('%s/%s/%s_%04d.dat', spmat_dir, datanames{i}, datanames{i}, datanums(i));
    sp = load(spfile) + 1;
    
    % read projection matrix
    [~, proj_sp] = create_mapping(sp,num_sp,sqrt(w_gloc.params.numNodes_crf),olddim);
    proj_crf = proj_sp;
    
    % projection matrices
    [proj_blk, ~] = create_mapping(sp,num_sp,sqrt(w_gloc.params.numNodes_rbm),olddim);
    proj_rbm = proj_blk;
    
    tS = tic;
    labelprob = inference_gloc(X, w_gloc, w_gloc.params, proj_crf, proj_rbm);
    tE = toc(tS);
    evaltime = evaltime + tE;
    
    [~, pred] = max(labelprob ,[], 1);
    err = sum(pred(:) ~= gt_splabels(:));
    tot_err = tot_err + err;
    tot_sp = tot_sp + num_sp;
    if verbose,
        fprintf('[%d/%d] err: %d/%d, acc = %g\n', i,length(datanames),err,num_sp,100*(1-tot_err/tot_sp));
    else
        if ~mod(i,10),
            fprintf('.');
        end
        if ~mod(i,100),
            fprintf('[%d/%d] ',i,length(datanames));
            fprintf('acc = %g\n',100*(1-tot_err/tot_sp));
        end
    end
    clear gt_splabels X sp proj_crf proj_rbm;
end
evaltime = evaltime/length(datanums);
acc = 100*(1-tot_err/tot_sp);

fprintf('acc = %g, inference time = %g (ex/sec)\n', acc, evaltime);

return