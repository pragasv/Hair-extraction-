%%% --- configuration for conditional RBM --- %%%
if ~exist('rmposfeat','var'),
    rmposfeat = 1;
end
if ~exist('verbose','var'),
    verbose = 0;
end

% spatial lr
if ~exist('dim_crf','var'),
    dim_crf = 16;
end
if ~exist('lrl2reg','var'),
    lrl2reg = 0.0001;
end
if ~exist('l2reg_node','var'),
    l2reg_node = lrl2reg;
end

% crbm
if ~exist('dim_rbm','var'),
    dim_rbm = 24;
end
if ~exist('opt_nodeup','var'),
    opt_nodeup = 0;
end
if ~exist('numHid','var'),
    numHid = 400;
end
if ~exist('l2reg','var'),
    l2reg = 0.0001;
end
if ~exist('epsilon','var'),
    epsilon = 0.003/2;
end
if ~exist('KCD','var'),
    KCD = 1;
end
if ~exist('anneal','var'),
    anneal = 1;
end
if ~exist('maxepoch','var'),
    maxepoch = 450;
end
if ~exist('batchSize','var'),
    batchSize = 0; % batch gradient descent
end


w_slr.params.dim = dim_crf;
w_slr.params.rmposfeat = rmposfeat;
w_slr.params.l2reg = lrl2reg;
