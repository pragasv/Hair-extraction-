% This script will train and evaluate conditional restricted Boltzmann machine
% for the task of Hair/Skin/Background labeling on LFW part label dataset
%
% input (see config_crbm.m for default values)
%   rmposfeat   : remove position features from superpixel features
%   verbose     : display progress during testing
%   dim_crf     : N (see reference)
%   lrl2reg     : weight decay for node weights
%   l2reg_node  : weight decay for node weights
%   dim_rbm     : R (see reference)
%   opt_nodeup  : node weight update if true
%   numHid      : number of hidden units, K (see reference)
%   l2reg       : weight decay for rbm weights
%   epsilon     : initial learning rate
%   KCD         : # of CD steps
%   maxepoch    : maximum number of iteration for gradient descent
%   batchSize   : size of minibatch (0 for batch gradient descent)
%   anneal      : annealing if true (initial KCD = 5, final KCD = 30)
%
% output
%   acc_train   : training accuracy
%   acc_valid   : validation accuracy
%   acc_test    : testing accuracy
%
%
% reference:
% Augmenting CRFs with Boltzmann Machine Shape Priors for Image Labeling, CVPR, 2013.
%

function [acc_train, acc_valid, acc_test] = run_lfw_gloc(rmposfeat,verbose,dim_crf,lrl2reg,l2reg_node,l2reg_edge,...
    dim_rbm,numHid,l2reg,epsilon,KCD,maxepoch,batchSize,anneal,maxepoch_gloc)


%%% --- default parameter values --- %%%
config_gloc;


%%% --- startup --- %%%
startup;
olddim = 250;   % original LFW image size
nlabel = 3;     % number of segmentation labels

load('sds_large.mat','sds');
load('esds_large.mat','esds');
if rmposfeat,
    sds(65:128) = [];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- spatial logistic regression --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/slr/');
fname_slr = sprintf('slr_l2r%g_rmposfeat%d_N%d',lrl2reg,rmposfeat,dim_crf);
train_lfw_slr;
save(sprintf('%s/%s.mat',fsave_dir,fname_slr),'w_slr');



% update save directory
fsave_dir = sprintf('%s/%s/',fsave_dir,fname_slr);
if ~exist(fsave_dir,'dir'),
    mkdir(fsave_dir);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- spatial conditional random field  --- %%%
%%% --- with mean-field inference         --- %%%
%%% --- train edge weights only           --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/scrf/');
fname_scrf = sprintf('scrf_edge_only_dim%d_l2n%g_l2e%g_rmposfeat%d',dim_crf,l2reg_node,l2reg_edge,rmposfeat);
skip_nodeup = 1;
train_lfw_scrf;
save(sprintf('%s/%s.mat',fsave_dir,fname_scrf),'w_scrf');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- conditional restricted Boltzmann machine --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/crbm/');
if batchSize == 0,
    batchSize = length(trainnames);
end
dataname = 'LFW';
opt_nodeup = 0;
fname_crbm = sprintf('crbm_%s_nD%d_nL%d_N%d_l2n%g_rbm_R%d_n%d_nH%d_l2r%g_eps%g_CD%d_ann%d_bS%d_iter%d',...
    dataname,size(w_slr.nodeWeights,1),nlabel,dim_crf,l2reg_node,dim_rbm,opt_nodeup,numHid,l2reg,epsilon,KCD,anneal,batchSize,maxepoch);
train_lfw_crbm;
fname = sprintf('%s/%s_done',fsave_dir,fname_crbm);
save(sprintf('%s.mat',fname),'w_rbm','w_slr','params');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- GLOC training --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/gloc/');

% gloc
anneal_gloc = 0;    % no annealing any more
KCD_gloc = 30;      % use CD steps that was used for pretraining of CRBM
epsilon_gloc = 2*params.epsilon/(1 + params.epsdecay*30); % multiply by 2 because we no more halve the weights
fname_gloc = sprintf('gloc_%s_nD%d_nL%d_N%d_l2n%g_l2e%g_rbm_R%d_nH%d_l2r%g_eps%.04g_CD%d_ann%d_bS%d',...
    dataname,size(w_slr.nodeWeights,1),nlabel,dim_crf,l2reg_node,l2reg_edge,dim_rbm,numHid,l2reg,epsilon_gloc,KCD_gloc,anneal_gloc,batchSize);
fname = sprintf('%s/%s_%04d',fsave_dir,fname_gloc,maxepoch_gloc);
train_lfw_gloc;
save(sprintf('%s.mat',fname),'w_gloc','params');



%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- evaluation --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=============================\n');
fprintf('Begin testing! (verbose:%d)\n',verbose);
fprintf('=============================\n\n');

acc_train = eval_lfw_gloc(w_gloc, trainnames, trainnums, sds, esds, verbose);
acc_valid = eval_lfw_gloc(w_gloc, validnames, validnums, sds, esds, verbose);
acc_test = eval_lfw_gloc(w_gloc, testnames, testnums, sds, esds, verbose);

fid = fopen(sprintf('%s/gloc.txt',log_dir),'a+');
fprintf(fid,'acc (val) = %g, acc (test) = %g, (%s)\n',acc_valid,acc_test,fname_gloc);
fclose(fid);

return;
