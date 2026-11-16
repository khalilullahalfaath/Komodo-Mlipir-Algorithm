addpath(pwd);
global FunctionID
A = readmatrix('points.csv');
out = zeros(size(A,1),1);
for r=1:size(A,1)
    FunctionID = A(r,1); n = A(r,2); x = A(r,4:3+n);
    out(r) = Evaluation(x);
end
writematrix([A(:,1) A(:,3) out], 'eval_result.csv');
