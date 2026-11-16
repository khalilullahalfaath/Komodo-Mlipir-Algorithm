addpath(pwd);
global FunctionID Dimension Nvar Ra Rb FthresholdFX
global PopSize MinAdaPopSize MaxAdaPopSize MaxNumEva
Dimension = 50; MaxNumEva = 25000; PopSize = 5;
MinAdaPopSize = PopSize*4; MaxAdaPopSize = PopSize*40;
NR = 30; res = [];
for fid = 1:23
    FunctionID = fid;
    for r = 1:NR
        rng(r);
        [Nvar,Ra,Rb,FthresholdFX] = GetFunction;
        Ra = ones(1,Nvar).*Ra; Rb = ones(1,Nvar).*Rb;
        PopSize = 5;
        [~,OptVal,NumEva,~,~,T,~] = KMA2D;
        res = [res; fid r OptVal NumEva T];
    end
    writematrix(res, 'kma_result.csv'); fprintf('F%d done\n', fid);
end
