%%% train spatial logistic regression on LFW labeling dataset
fprintf('\nstart SLR training\n\n');
fprintf('processing the features!!\n');


% load label, superpixels, features
tr_label = cell(length(trainnames),1);
tr_feat = cell(length(trainnames),1);
tr_proj = cell(length(trainnames),1);
for i = 1:length(trainnames),
    if ~mod(i,10), fprintf('.'); end
    if ~mod(i,500), fprintf('%d\n',i); end
    
    % load full data
    gt_casename = sprintf('%s/%s/%s_%04d.dat', gt_dir, trainnames{i}, trainnames{i}, trainnums(i));
    gt_case = load(gt_casename);
    gt_case = gt_case + 1;
    gt_splabels = gt_case(2:end);   % the first value is the number of nodes
    tr_label{i} = gt_splabels;
    
    % read superpixel features
    [~, H , ~, ~] = getFeatures(trainnames{i}, trainnums(i), features_dir);
    [~, num_sp] = size(H);
    if rmposfeat,
        H(65:128,:) = [];
    end
    whiteH = H ./ repmat(sds, [1, num_sp]); % whiten the features
    feat = whiteH';
    numFeat = size(feat, 1);
    feat = [feat ones(numFeat, 1)]; % add bias term
    tr_feat{i} = feat;
    
    % read superpixel data
    spfile = sprintf('%s/%s/%s_%04d.dat', spmat_dir, trainnames{i}, trainnames{i}, trainnums(i));
    sp = load(spfile) + 1;
    
    % create projection matrix
    [~, proj_sp] = create_mapping(sp,num_sp,dim_crf,olddim);
    tr_proj{i} = proj_sp;
    clear gt_splabels sp feat proj_sp;
end
fprintf('\ndone!\n');

% superpixel-level label data
k = 0;
for i = 1:length(tr_feat),
    tr_label_patch(k+1:k+size(tr_feat{i},1)) = tr_label{i};
    k = k + size(tr_feat{i},1);
end
tr_label_patch = multi_output(tr_label_patch,nlabel);
clear tr_label;

options.Method = 'lbfgs';
options.maxIter = 1500;
options.MaxFunEvals = 1500;
options.display = 'on';

num_in = size(tr_feat{1},2);
num_out = nlabel;

W = 0.1*randn(dim_crf^2,num_in,num_out);
theta = W(:);

%%% --- train spatial logistic regression --- %%%
[opttheta, ~] = minFunc( @(p) cost_spatial_lr(p,tr_feat,tr_label_patch,tr_proj,num_in,num_out,lrl2reg), theta, options);
clear tr_feat tr_proj tr_label_patch W;
w_slr.nodeWeights = permute(reshape(opttheta,dim_crf^2,num_in,num_out),[2 3 1]);

