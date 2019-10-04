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

function [acc_train, acc_valid, acc_test] = run_lfw_crbm(rmposfeat,verbose,dim_crf,lrl2reg,l2reg_node,...
    dim_rbm,opt_nodeup,numHid,l2reg,epsilon,KCD,maxepoch,batchSize,anneal)

%%% --- default parameter values --- %%%
config_crbm;


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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- conditional restricted Boltzmann machine --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/crbm/');
if batchSize == 0,
    batchSize = length(trainnames);
end
dataname = 'LFW';
fname_crbm = sprintf('crbm_%s_nD%d_nL%d_N%d_l2n%g_rbm_R%d_n%d_nH%d_l2r%g_eps%g_CD%d_ann%d_bS%d_iter%d',...
    dataname,size(w_slr.nodeWeights,1),nlabel,dim_crf,l2reg_node,dim_rbm,opt_nodeup,numHid,l2reg,epsilon,KCD,anneal,batchSize,maxepoch);
train_lfw_crbm;
fname = sprintf('%s/%s_done',fsave_dir,fname_crbm);
save(sprintf('%s.mat',fname),'w_rbm','w_slr','params');



%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- evaluation --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=============================\n');
fprintf('Begin testing! (verbose:%d)\n',verbose);
fprintf('=============================\n\n');

acc_train = eval_lfw_crbm(w_rbm, w_slr, params, trainnames, trainnums, sds, verbose);
acc_valid = eval_lfw_crbm(w_rbm, w_slr, params, validnames, validnums, sds, verbose);
acc_test = eval_lfw_crbm(w_rbm, w_slr, params, testnames, testnums, sds, verbose);

fid = fopen(sprintf('%s/crbm.txt',log_dir),'a+');
fprintf(fid,'acc (val) = %g, acc (test) = %g, (%s)\n',acc_valid,acc_test,fname_crbm);
fclose(fid);

return;
