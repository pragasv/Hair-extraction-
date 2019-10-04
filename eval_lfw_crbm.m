function [acc, evaltime] = eval_lfw_crbm(w_rbm, w_scrf, params, datanames, datanums, sds, verbose)
startup_directory;
olddim = 250;
addpath('model/crbm');

w_rbm.vishidrs = reshape(w_rbm.vishid,size(w_rbm.vishid,1)*size(w_rbm.vishid,2),size(w_rbm.vishid,3));
w_rbm.visbiasrs  = reshape(w_rbm.visbiases,size(w_rbm.vishid,1)*size(w_rbm.vishid,2),1);
w_scrf.nodeWeightsrs = reshape(w_scrf.nodeWeights,size(w_scrf.nodeWeights,1)*size(w_scrf.nodeWeights,2),size(w_scrf.nodeWeights,3));
w_rbm.vishidperm = reshape(w_rbm.vishid,size(w_rbm.vishid,1),size(w_rbm.vishid,2)*size(w_rbm.vishid,3));

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
    [numNodes, H] = getFeatures(datanames{i}, datanums(i), features_dir);
    if w_scrf.params.rmposfeat,
        H(65:128, :) = [];
    end
    % node features
    H = bsxfun(@rdivide,H,sds);
    H(end+1,:) = 1; % add bias term
    
    X = struct('numNodes', numNodes, 'nodeFeatures', {H});
    num_sp = numNodes;
    
    % read superpixel data
    spfile = sprintf('%s/%s/%s_%04d.dat', spmat_dir, datanames{i}, datanames{i}, datanums(i));
    sp = load(spfile) + 1;
    
    % read projection matrix
    [~, proj_sp] = create_mapping(sp,num_sp,sqrt(params.numNodes_crf),olddim);
    proj_crf = proj_sp;
    
    % projection matrices
    [proj_blk, ~] = create_mapping(sp,num_sp,sqrt(params.numNodes_rbm),olddim);
    proj_rbm = proj_blk;
    
    tS = tic;
    labelprob = inference_crbm(X, w_rbm, w_scrf, params, proj_crf, proj_rbm, 2);
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