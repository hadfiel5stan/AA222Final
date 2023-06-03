using Random
using NonlinearSolve, StaticArrays

function uniform_proj_plan(m, n)
    ps = [randperm(m) for i in 1:n]
    [[ps[i][j] for i in 1:n] for j in 1:m]
end

function fitlogistic(u, p)
    L = u[1]
    k = u[2]
    x0 = u[3]

    # ret1 = L/(1 + exp(k*(0 - x0))) - p[1];
    ret1 = 0;
    ret2 = L*x0 - L*(0 - log(exp(k*(0 - x0)) + 1)/k) - p[2];
    ret3 = L*x0 - L*(0.5 - log(exp(k*(0.5 - x0)) + 1)/k) - p[3];

    return [ret1, ret2, ret3]
end 

u0 = @SVector[1, 1, 0.5]
p = [10, 1, 0.2]

probN = NonlinearProblem(fitlogistic, u0, p)
solver = solve(probN, Klement(), reltol=1e-9)
