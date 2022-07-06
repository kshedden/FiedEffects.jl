
function REModel(X, y, clust::AbstractVector)
    @assert length(y) == size(X, 1)

    clx = [indx(g) for g in clust]
    n, p = size(X)
    return REModel(X, y, zeros(n), zeros(p), zeros(n), clx)
end

function updateResid!(m::REModel)
    m.resid .= m.y - m.Xbeta
end

# The log-likelihood function
function loglike(m::REModel, beta::AbstractVector)::Float64
    m.Xbeta .= m.X * beta
    updateResid!(m)
    ll = -sum(abs2, m.resid) / 2
    return ll
end

# The gradient of the log-likelihood function.
function grad(m::REModel, beta::AbstractVector)
    m.Xbeta .= m.X * beta
    updateResid!(m)
    g = m.X' * m.resid
    return g
end

function hess(m::REModel, beta::AbstractVector)
    return Symmetric(m.X' * m.X)
end

function StatsBase.fit!(m::REModel)

    m0 = lm(m.X, m.y)
    m.beta .= coef(m0)
    updateResid!(m)
end

coef(m::REModel) = m.beta

function StatsBase.fit(
    ::Type{REModel},
    X::AbstractMatrix,
    y::AbstractVector,
    clx::AbstractVector,
)
    m = REModel(X, y, clx)
    fit!(m)
    return m
end

randomeffects(F, D, args...; kwargs...) = fit(REModel, F, D, args...; kwargs...)
