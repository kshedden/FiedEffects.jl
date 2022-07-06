using Statistics, Optim, LinearAlgebra, Distributions, Printf

# Returns a list of indices corresponding to the positions in g with each distinct value.
function indx(g::AbstractVector)

    u = unique(g)
    sort!(u)

    # Map labels to index positions
    um = Dict{eltype(g),Int}()
    for (i, v) in enumerate(u)
        um[v] = i
    end

    ix = [Int[] for _ in eachindex(u)]
    for (i, v) in enumerate(g)
        push!(ix[um[v]], i)
    end

    return ix
end

function FEModel(X, y, fe::AbstractVector, clust::AbstractVector)
    @assert length(y) == size(X, 1)

    fex = [indx(g) for g in fe]
    clx = [indx(g) for g in clust]
    n, p = size(X)
    nfe = length(fe) # number of fixed effects variables
    FE = zeros(n, nfe)
    return FEModel(X, y, zeros(n), zeros(p), zeros(n), fex, FE, clx)
end

function updateResid!(m::FEModel)
    m.resid .= m.y - m.Xbeta
    for j = 1:size(m.FE, 2)
        m.resid .-= m.FE[:, j]
    end
end

# One step update of the fixed effects.
function iterFE!(m::FEModel)

    updateResid!(m)

    # Loop over fixed effects variables
    for c in eachindex(m.fex)

        # Loop over levels within each fixed effects variable
        for j in eachindex(m.fex[c])
            ii = m.fex[c][j]
            mn = mean(m.resid[ii])
            m.FE[ii, c] .+= mn
            m.resid[ii] .-= mn
        end
    end
end

# Update the fixed effects estimates for the current beta.
function updateFE!(m::FEModel; maxit::Int = 100, tol::Float64 = 1e-4)

    m.FE .= 0
    n = length(m.y)
    nfe = size(m.FE, 2)
    FE1 = zeros(n, nfe)
    FE2 = zeros(n, nfe)
    FE3 = zeros(n, nfe)
    DF1 = zeros(n, nfe)
    DF2 = zeros(n, nfe)
    DFX = zeros(n, nfe)

    for itr = 1:maxit
        # Iterate twice naively
        FE1 .= m.FE
        iterFE!(m)
        FE2 .= m.FE
        iterFE!(m)
        FE3 .= m.FE

        # First differences
        DF1 .= FE2 - FE1
        DF2 .= FE3 - FE2

        # Second differences
        DFX .= DF2 - DF1

        # Irons and Tuck update
        for j = 1:nfe
            m.FE[:, j] .-= dot(DF2[:, j], DFX[:, j]) * DF2[:, j] / sum(abs2, DFX[:, j])
        end

        if norm(DF1) + norm(DF2) < tol
            break
        end
    end
end

# The concentrated log-likelihood function
function loglike(m::FEModel, beta::AbstractVector)::Float64
    m.Xbeta .= m.X * beta
    updateFE!(m)
    updateResid!(m)
    ll = -sum(abs2, m.resid) / 2
    return ll
end

# The gradient of the concentrated log-likelihood function.
function grad(m::FEModel, beta::AbstractVector)
    m.Xbeta .= m.X * beta
    updateFE!(m)
    updateResid!(m)
    g = m.X' * m.resid
    return g
end

function StatsBase.fit!(m::FEModel)

    # Special case does not require fixed effects estimation.
    if length(m.fex) == 0
        @warn("Fixed effects model contains no fixed effects")
        return
    end

    function g!(gr, beta)
        gr .= -grad(m, beta)
    end

    # Starting value
    m0 = lm(m.X, m.y)
    beta0 = coef(m0)

    gr = grad(m, m.beta)

    f = beta -> -loglike(m, beta)

    r = optimize(f, g!, beta0, LBFGS())

    if !Optim.converged(r)
        @warn("Did not converge")
    end

    m.beta .= Optim.minimizer(r)
end

"""
Return the estimated fixed effects.
"""
function fe_estimates(m::FEModel)

    updateFE!(m)
    fe = Vector{Vector{Float64}}()
    for c in eachindex(m.fex)
        v = []
        for j in eachindex(m.fex[c])
            ii = m.fex[c][j]
            push!(v, first(m.FE[ii, c]))
        end
        push!(fe, v)
    end

    return fe
end

function dsdb(m::FEModel, s0::AbstractMatrix)
    nfe = length(m.fex)
    n, p = size(m.X)
    s0 = copy(s0)
    inc = zeros(n, p)
    for c = 1:nfe
        inc .= 0
        for j in eachindex(m.fex[c])
            ii = m.fex[c][j]
            inc[ii, :] .= -mean(s0[ii, :] + m.X[ii, :], dims = 1)
        end
        s0 .+= inc
    end
    return s0
end

function hess(m::FEModel, beta::AbstractVector; maxit::Int = 100, tol::Float64 = 1e-4)

    n, p = size(m.X)
    nfe = length(m.fex)
    s0 = zeros(n, p)
    s1 = zeros(n, p)
    s2 = zeros(n, p)
    ds1 = zeros(n, p)
    ds2 = zeros(n, p)
    dsx = zeros(n, p)
    X = m.X

    m.Xbeta .= m.X * beta
    updateFE!(m)
    updateResid!(m)

    # First do a few iterations without Irons-Tuck.
    for it = 1:5
        s0 = dsdb(m, s0)
    end

    # Irons-Tuck iterations.
    for it = 1:maxit

        # Iterate twice naively
        s1 .= dsdb(m, s0)
        s2 .= dsdb(m, s1)

        # First differences
        ds1 .= s1 - s0
        ds2 .= s2 - s1

        # Second differences
        dsx .= ds2 - ds1

        for j = 1:p
            f = sum(abs2, dsx[:, j])
            if f < 1e-6
                # Greedy update
                s0[:, j] .= s2[:, j]
            else
                # Irons and Tuck update
                s0[:, j] .= s2[:, j] - dot(ds2[:, j], dsx[:, j]) * ds2[:, j] / f
            end
        end

        if sqrt(sum(abs2, dsx)) < tol
            break
        end
    end

    u = m.X + s0
    he = u' * u

    return Symmetric(he)
end

# Get all distinct pairs of observations that fall into the same group of any cluster.
function pclust(m::FRModel)

    ix = Set{Tuple{Int,Int}}()

    # Always add the diagonal terms
    for i in eachindex(m.y)
        push!(ix, tuple(i, i))
    end

    # These are the off-diagonal terms
    for j in eachindex(m.clx)
        for k in eachindex(m.clx[j])
            v = m.clx[j][k]
            for i1 in eachindex(v)
                for i2 = 1:i1-1
                    v1, v2 = v[i1], v[i2]
                    v1, v2 = if v1 < v2
                        v1, v2
                    else
                        v2, v1
                    end
                    push!(ix, tuple(v1, v2))
                end
            end
        end
    end

    return ix
end

function vcov(m::FEModel; naive::Bool = false)
    return vcov_helper(m, naive)
end

function vcov(m::REModel; naive::Bool = false)
    return vcov_helper(m, naive)
end

function vcov_helper(m::FRModel, naive::Bool)

    hm = hess(m, m.beta)
    hmi = pinv(hm)
    a, b = eigen(hmi)
    if minimum(a) < -1e-10
        # This should never happen
        @warn(
            @sprintf(
                "Hessian is not positive definite, minimum eigenvalue is %f",
                minimum(a)
            )
        )
        a = clamp.(a, 0, Inf)
        hmi = b * diagm(a) * b'
    end
    n, p = size(m.X)

    if naive
        scale = sum(abs2, m.resid) / (n - p)
        return scale * hmi
    end

    ix = pclust(m)
    X = m.X

    vm = zeros(p, p)
    u = m.resid
    for (i1, i2) in ix
        vm .+= u[i1] * u[i2] * X[i1, :] * X[i2, :]'
        if i1 != i2
            vm .+= u[i1] * u[i2] * X[i2, :] * X[i1, :]'
        end
    end
    vm = Symmetric(vm)

    a, b = eigen(vm)
    if minimum(a) < 0
        a = clamp.(a, 0, Inf)
        vm = b * diagm(a) * b'
    end

    return hmi * vm * hmi
end

coef(m::FEModel) = m.beta

function StatsBase.coeftable(mm::FRModel; level::Real = 0.95)
    cc = coef(mm)
    se = sqrt.(diag(vcov(mm)))
    zz = cc ./ se
    pv = 2 * ccdf.(Ref(Normal()), abs.(zz))
    ci = se * quantile(Normal(), (1 - level) / 2)
    levstr = isinteger(level * 100) ? string(Integer(level * 100)) : string(level * 100)
    na = ["x$i" for i in eachindex(mm.beta)]
    CoefTable(
        hcat(cc, se, zz, pv, cc + ci, cc - ci),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $levstr%", "Upper $levstr%"],
        na,
        4,
        3,
    )
end

function StatsBase.fit(
    ::Type{FEModel},
    X::AbstractMatrix,
    y::AbstractVector,
    fex::AbstractVector,
    clx::AbstractVector,
)
    m = FEModel(X, y, fex, clx)
    fit!(m)
    return m
end

fixedeffects(F, D, args...; kwargs...) = fit(FEModel, F, D, args...; kwargs...)
