module FixedEffects

using GLM, StatsBase, StatsModels

import StatsBase: coef, coeftable, vcov, fit

export FRModel, FEModel, fixedeffects, fe_estimates, fit!, fit, coef, vcov
export REModel, randomeffects

include("defs.jl")
include("fixef.jl")
include("ranef.jl")

end
