abstract type FRModel <: GLM.LinPredModel end

# A regression model that has at least one fixed effect, and may have non-independent,
# heteroscedastic residual variation.  The model is fit using a concentrated likelihood.
abstract type FixedEffectsModel <: FRModel end

# A regression model that has heteroscedastic residual variation.
abstract type RandomEffectsModel <: FRModel end

struct FEModel <: FixedEffectsModel

    # Covariates
    X::AbstractMatrix

    # Response values
    y::AbstractVector

    # Fitted value
    Xbeta::AbstractVector

    # Current parameter estimates
    beta::AbstractVector

    # Current residuals
    resid::AbstractVector

    # fex[j][k] contains indices for all observations with level k of fixed effects variable j.
    fex::Vector{Vector{Vector{Int}}}

    # Column of j of FE is the total of fixed effects for variable j.
    FE::AbstractMatrix

    # clx[j][k] contains indices for all observations with level k of clustering variable j.
    clx::Vector{Vector{Vector{Int}}}
end

struct REModel <: RandomEffectsModel

    # Covariates
    X::AbstractMatrix

    # Response values
    y::AbstractVector

    # Fitted value
    Xbeta::AbstractVector

    # Current parameter estimates
    beta::AbstractVector

    # Current residuals
    resid::AbstractVector

    # clx[j][k] contains indices for all observations with level k of clustering variable j.
    clx::Vector{Vector{Vector{Int}}}
end

StatsModels.drop_intercept(::Type{<:FEModel}) = true
