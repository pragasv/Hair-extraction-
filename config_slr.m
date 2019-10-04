%%% --- configuration for spatial logistic regression --- %%%
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

w_slr.params.dim = dim_crf;
w_slr.params.rmposfeat = rmposfeat;
w_slr.params.l2reg = lrl2reg;
