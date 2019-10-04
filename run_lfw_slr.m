% This script will train and evaluate a spatial logistic regression
% for the task of Hair/Skin/Background labeling on LFW part label dataset
%
% input (see config_slr.m for default values)
%   rmposfeat   : remove position features from superpixel features
%   verbose     : display progress during testing
%   dim_crf     : N (see reference)
%   lrl2reg     : weight decay for node weights
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

function [acc_train, acc_valid, acc_test] = run_lfw_slr(rmposfeat,verbose,dim_crf,lrl2reg)

%%% --- default parameter values --- %%%
config_slr;


%%% --- startup --- %%%
startup;
olddim = 250;   % original LFW image size
nlabel = 3;     % number of segmentation labels

load('sds_large.mat','sds');
load('esds_large.mat','esds');
if rmposfeat,
    % remove position features
    sds(65:128) = [];
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- spatial logistic regression --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('model/slr/');
fname_slr = sprintf('slr_l2r%g_rmposfeat%d_N%d',lrl2reg,rmposfeat,dim_crf);
train_lfw_slr;
save(sprintf('%s/%s.mat',fsave_dir,fname_slr),'w_slr');



%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% --- evaluation --- %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n=============================\n');
fprintf('Begin testing! (verbose:%d)\n',verbose);
fprintf('=============================\n\n');

acc_train = eval_lfw_slr(w_slr, trainnames, trainnums, sds, verbose);
acc_valid = eval_lfw_slr(w_slr, validnames, validnums, sds, verbose);
acc_test = eval_lfw_slr(w_slr, testnames, testnums, sds, verbose);

fid = fopen(sprintf('%s/slr.txt',log_dir),'a+');
fprintf(fid,'acc (val) = %g, acc (test) = %g, (%s)\n',acc_valid,acc_test,fname_slr);
fclose(fid);

return;
