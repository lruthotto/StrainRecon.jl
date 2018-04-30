using StrainRecon
using Base.Test
# include("../src/asqp.jl")
@testset "ASQP" begin

print("\t\tsolve 2D QP without equality constraints...\n\t\t")
n = 2

G = eye(n)
c = 0*[1;1.]

Ae = zeros(0,n)
be = zeros(0)
me = size(Ae,1)

Ai = eye(n)
bi = 0*rand(n)
mi = size(Ai,1)

x0  = -1*rand(n)

xopt,iter,Wc = asqp(G,c, Ae, be, Ai, bi, x0) # start from infeasible x0 .< 0
@test all((Ai*xopt-bi).>=0)

x0 = rand(n) + 1
xopt,iter,Wc = asqp(G,c, Ae, be, Ai, bi, x0) # start from feasible x0 .> 1
@test all((Ai*xopt-bi).>=0)


# x* = [1;0]
Ai = [1 0.0]
G = eye(2);
c = zeros(2,1);
bi = [1.0]
Ae = zeros(0,2)
be = zeros(0)
xopt, iter, Wc = asqp(G,c,Ae,be,Ai,bi) # start from infeasible x0 = [0;0]
@test all(xopt.==[1;0])
@test all(Ai*xopt-bi.>=0)

x0 = [1;2]
xopt, iter, Wc = asqp(G,c,Ae,be,Ai,bi,x0) # start from feasible x0 = [1;2]
@test all(xopt.==[1;0])
@test all(Ai*xopt-bi.>=0)

# x* = [10, 8]
G = [1 -1; -1 2]
c = [-2; -6]
A = [1 1; -1 2; 2 1]
b = [2; 2; 3]
Ae = zeros(0,2)
be = zeros(0)
xopt, iter, Wc = asqp(G,c,Ae,be,A,b) # start from infeasible x0 = [0;0]
@test all((A*xopt-b).>=0)
@test all(xopt .== [10;8])

x0 = [1;2]
xopt, iter, Wc = asqp(G,c,Ae,be,A,b,x0) # start from feasible x0 = [1;2]
@test all((A*xopt-b).>=0)
@test all(xopt .== [10;8])


# x* = [3;5;7], unconstrained example, G is positive definite, global min exists
G = [5 -2 -1; -2 4 3; -1 3 5]
A = zeros(3,3)
c = [2; -35; -47]
b = [0; 0; 0]
Ae = zeros(0,3)
be = zeros(0)

xopt, iter, Wc = asqp(G,c,Ae,be,A,b)
@test all((A*xopt-b).>=0)
@test norm(xopt - [3.0;5.0;7.0])/norm([3.;5;7]) < 1e-14 # xopt = x* but comparison fails

print("\t\tsolve 2D QP with inequality and equality constraints...\n\t\t")

# x* = [0.5;0.5]

n = 2

G = eye(n)
c = 0*[1;1.]

Ae = [1 1]
be = ones(1)

Ai = eye(n)
bi = 0.0*ones(n)

x0  = [-0.5;0.5]

xopt,iter,Wc = asqp(G,c, Ae, be, Ai, bi, x0) # x0 is infeasible for both equality and inequality constraints
@test all(Ai*xopt.>=bi)
@test norm(Ae*xopt-be) < 1e-14

x0  = -1*rand(n)
xopt,iter,Wc = asqp(G,c, Ae, be, Ai, bi, x0) # x0 is infeasible for both equality and inequality constraints
@test all(Ai*xopt.>=bi)
@test norm(Ae*xopt-be) < 1e-14 # xopt not feasible for equality constraints

x0  = rand(n)+1
xopt,iter,Wc = asqp(G,c, Ae, be, Ai, bi, x0) # x0 is feasible for only inequality constraints
@test all(Ai*xopt.>=bi)
@test norm(Ae*xopt-be) < 1e-14 # xopt not feasible for equality constraints


    # x* = [0.25;0.75]
    G = [4 1;1 2];
    Ai = [1 0;0 1];
    bi = [0;0]
    Ae = [1 1];
    c = [1; 1];
    be = [1];

    x0 = 0*c
    xopt, iter, Wc = asqp(G,c,Ae,be,Ai,bi,x0) # x0 is feasible for only inequality constraint
    @test all(Ai*xopt.>=bi)
    @test norm(Ae*xopt-be) < 1e-14

    x0 = [2;-1]
    xopt, iter, Wc = asqp(G,c,Ae,be,Ai,bi,x0) # x0 is feasible for only equality constraint
    @test all(Ai*xopt.>=bi)
    @test norm(Ae*xopt-be) < 1e-14

    x0 = -1*rand(2)
    xopt, iter, Wc = asqp(G,c,Ae,be,Ai,bi,x0) # x0 is infeasible for inequality and equality constraints
    @test all(Ai*xopt.>=bi)
    @test norm(Ae*xopt-be) < 1e-14

    x0 = [1e-3;1-1e-3]
    xopt, iter, Wc = asqp(G,c,Ae,be,Ai,bi,x0) # x0 is feasible for both inequality and equality constraints
    @test all(Ai*xopt.>=bi)
    @test norm(Ae*xopt-be) < 1e-14


end
