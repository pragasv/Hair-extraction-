%%% train logistic regression on LFW labeling dataset
fprintf('\nstart LR training\n\n');
fprintf('processing the features!!\n');


tr_feat = [];
tr_label = [];
for i = 1:length(trainnames),
    if ~mod(i,10), fprintf('.'); end
    if ~mod(i,500), fprintf('%d\n',i); end
    
    % load full data
    gt_casename = sprintf('%s/%s/%s_%04d.dat', gt_dir, trainnames{i}, trainnames{i}, trainnums(i));
    gt_case = load(gt_casename);
    gt_case = gt_case + 1;         % increase offset by 1
    gt_splabels = gt_case(2:end);   % the first value is the number of nodes
    tr_label = [tr_label multi_output(gt_splabels,nlabel)];
    
    % read superpixel features
    [~, H , ~, ~] = getFeatures(trainnames{i}, trainnums(i), features_dir);
    [~, num_sp] = size(H);
    if rmposfeat,
        H(65:128,:) = [];
    end
    whiteH = H ./ repmat(sds, [1, num_sp]); % whiten the features
    feat = whiteH';
    numFeat = size(feat, 1);
    feat = [feat ones(numFeat, 1)];
    tr_feat = [tr_feat feat'];
    
    clear gt_splabels feat;
end

options.Method = 'lbfgs';
options.maxIter = 1500;
options.MaxFunEvals = 1500;
options.display = 'on';

num_in = size(tr_feat,1);
num_out = nlabel;

W = 0.1*randn(num_in,num_out);
theta = W(:);

% train spatial logistic regression
[opttheta, ~] = minFunc( @(p) cost_lr(p,tr_feat,tr_label,num_in,num_out,lrl2reg), theta, options);

w_lr.nodeWeights = reshape(opttheta,num_in,num_out);
