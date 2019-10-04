%%% --- configuration for spatial CRF --- %%%
if ~exist('rmposfeat','var'),
    rmposfeat = 1;
end
if ~exist('verbose','var'),
    verbose = 0;
end

% spatial crf
if ~exist('dim_crf','var'),
    dim_crf = 16;
end
if ~exist('lrl2reg','var'),
    lrl2reg = 0.00003;
end
if ~exist('l2reg_node','var'),
    l2reg_node = lrl2reg;
end
if ~exist('l2reg_edge','var'),
    l2reg_edge = 0;
end
if ~exist('skip_nodeup','var'),
    skip_nodeup = 0;
end

w_scrf.params.rmposfeat = rmposfeat;
w_scrf.params.l2reg_node = l2reg_node;
w_scrf.params.l2reg_edge = l2reg_edge;
w_scrf.params.dim = dim_crf;
