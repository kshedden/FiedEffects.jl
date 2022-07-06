using Test,
    FixedEffects,
    StableRNGs,
    FiniteDifferences,
    Statistics,
    UnicodePlots,
    GLM,
    LinearAlgebra,
    DataFrames

@testset "Check setup" begin

    rng = StableRNG(123)
    y = [5, 6, 7, 8]
    X = randn(rng, 4, 2)
    fe = [[1, 1, 2, 2], [1, 2, 1, 2]]
    cl = [[1, 1, 2, 2]]

    fe = FEModel(X, y, fe, cl)
    re = REModel(X, y, cl)

    @test isapprox(fe.fex, [[[1, 2], [3, 4]], [[1, 3], [2, 4]]])
    @test isapprox(fe.clx, [[[1, 2], [3, 4]]])
    @test isapprox(re.clx, [[[1, 2], [3, 4]]])
end

@testset "Check random effects model" begin

    rng = StableRNG(123)
    n, p = 4000, 3
    X = randn(rng, n, p)
    m = 20 # group size
    q = div(n, m) # number of groups
    u = randn(q)
    clx = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]
    e = randn(rng, q)[clx[1]] + randn(n)
    y = X[:, 1] - X[:, 2] + e

    df = DataFrame(:y => y)
    for j = 1:p
        df[:, Symbol("x$(j)")] = X[:, j]
    end

    fml = @formula(y ~ x1 + x2 + x3)
    m = randomeffects(fml, df, clx)

    @test isapprox(coef(m), Float64[0, 1, -1, 0], atol = 0.1, rtol = 0.1)

    # TODO check vcov here
    c = vcov(m)
end

@testset "Check estimated fixed effects" begin

    rng = StableRNG(132)
    n = 10000
    p = 5
    nfe = 2
    m = 20 # group size
    q = div(n, m) # number of groups
    fex = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]
    X = randn(rng, n, p)
    X[:, 1] .= 1
    y = randn(rng, n)
    fet = Vector{Vector{Float64}}()
    for j = 1:nfe
        v = randn(rng, q)
        push!(fet, v)
        y .+= v[fex[j]]
    end
    m = FEModel(X, y, fex, [])
    FixedEffects.updateFE!(m)

    fee = fe_estimates(m)

    for i in eachindex(fet)
        @test cor(fet[i], fee[i]) > 0.9

        xx = ones(length(fet[i]), 2)
        xx[:, 2] = fet[i]
        mm = lm(xx, fee[i])
        cc = coef(mm)
        @test isapprox(cc, Float64[0, 1], atol = 0.1, rtol = 0.05)
    end
end

@testset "Check gradient and Hessian for fixed effects model" begin

    rng = StableRNG(123)
    n = 1000
    p = 5
    nfe = 2
    m = 20
    q = div(n, m)
    fex = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]
    X = randn(rng, n, p)
    X[:, 1] .= 1
    y = randn(rng, n)
    for j = 1:nfe
        v = randn(rng, q)
        y .+= v[fex[j]]
    end
    m = FEModel(X, y, fex, [])

    for k = 1:10
        beta = randn(p)
        gr = FixedEffects.grad(m, beta)
        f = x -> FixedEffects.loglike(m, x)
        ngr = grad(central_fdm(10, 1), f, beta)[1]
        @test isapprox(gr, ngr, atol = 1e-4, rtol = 1e-4)

        he = FixedEffects.hess(m, beta)
        nhe = jacobian(central_fdm(5, 1), x -> -FixedEffects.grad(m, x), beta)[1]
        @test isapprox(he, nhe, atol = 1e-4, rtol = 1e-4)
    end
end

@testset "Check estimated parameters for fixed effects model" begin
    rng = StableRNG(123)
    n = 10000
    p = 5
    ncfe = 2
    m = 50
    q = div(n, m)
    fex = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]
    X = randn(rng, n, p)
    X[:, 1] .= 1
    y = X[:, 2] - X[:, 3] + randn(rng, n)
    fet = Vector{Vector{Float64}}()
    for j = 1:ncfe
        v = randn(rng, q)
        push!(fet, v)
        y .+= v[fex[j]]
    end
    m = FEModel(X, y, fex, [])
    fit!(m)
    @test isapprox(coef(m), Float64[0, 1, -1, 0, 0], rtol = 0.1, atol = 0.1)
end

@testset "Check estimated variance/covariance matrix for fixed effects model" begin
    rng = StableRNG(123)
    n = 1000
    p = 5
    ncfe = 2
    m = 50
    q = div(n, m)
    fex = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]
    X = randn(rng, n, p)
    y = X[:, 2] - X[:, 3] + randn(rng, n)
    fet = Vector{Vector{Float64}}()
    for j = 1:ncfe
        v = randn(rng, q)
        push!(fet, v)
        y .+= v[fex[j]]
    end
    m = FEModel(X, y, fex, fex)
    fit!(m)
    na = vcov(m; naive = true)
    rb = vcov(m; naive = false)
    @test minimum(diag(na)) .> 0
    @test minimum(diag(rb)) .> 0
end

@testset "Check formula" begin
    rng = StableRNG(123)
    n = 1000
    p = 5
    m = 50
    q = div(n, m)
    dx = DataFrame(:x1 => randn(n), :x2 => randn(n))
    dx[:, :y] = dx[:, :x1] + dx[:, :x2] + randn(n)
    fex = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]
    r = fixedeffects(@formula(y ~ x1 + x2), dx, fex, fex)

    X = ones(n, 2)
    X[:, 1] = dx[:, :x1]
    X[:, 2] = dx[:, :x2]
    y = dx[:, :y]
    m = FEModel(X, y, fex, fex)
    fit!(m)

    @test isapprox(coef(r), coef(m))
    @test isapprox(vcov(r), vcov(m))
end
