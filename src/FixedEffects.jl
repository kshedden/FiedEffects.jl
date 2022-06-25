module FixedEffects

using GLM, StatsBase, StatsModels

import StatsBase: coef, coeftable, vcov, fit

export FEModel, fixedeffects, fit!, fit, coef, vcov

include("fixef.jl")

end
