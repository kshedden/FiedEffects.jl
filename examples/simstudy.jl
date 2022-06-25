using FixedEffects, LinearAlgebra, Statistics, Printf

function gclust(fex, r)
    n = length(first(fex))
    y = zeros(n)
    for j in eachindex(fex)
        v = randn(maximum(fex[j]))
        y .+= v[fex[j]] + r * randn(n)
    end
    return y
end

# If mode == 1, the same grouping variables are used for fixed effects and
# clustering.  If mode == 2, different grouping variables are used for
# fixed effects and clustering.
function gendat(fixef, clust, mode)
    n = 2000
    p = 2
    m = 20 # group size
    q = div(n, m) # number of groups

    # Two grouping variables
    grp = [kron(ones(Int, m), 1:q), kron(1:q, ones(Int, m))]

    # Two more grouping variables
    grpr = if mode == 1
        grp
    else
        grpr = []
        for j = 1:2
            ii = sortperm(rand(n))
            push!(grpr, grp[j][ii])
        end
        grpr
    end

    # Design matrix
    X = zeros(n, p)
    for j = 1:p
        X[:, j] = gclust(vcat(grp, grpr), 1)
    end

    y = zeros(n)
    y .+= gclust(grpr, 0)
    y .+= randn(n)

    fex = fixef ? grp : []
    clx = if clust
        mode == 1 ? grp : grpr
    else
        []
    end

    m = FEModel(X, y, fex, clx)
    fit!(m)
    return m
end

function run(fixef, clust, mode)
    p = 2
    nrep = 500
    zs = zeros(nrep, p)
    zn = zeros(nrep, p)
    bb = zeros(p)
    for i = 1:nrep
        m = gendat(fixef, clust, mode)
        b = coef(m)
        bb .+= b
        s = sqrt.(diag(vcov(m; naive = false)))
        zs[i, :] = b ./ s
        s = sqrt.(diag(vcov(m; naive = true)))
        zn[i, :] = b ./ s
    end
    bb ./= nrep
    println("  Mean coef: ", bb)
    println("  Mean of naive Z-scores: ", mean(zn, dims = 1))
    println("  Mean of robust Z-scores: ", mean(zs, dims = 1))
    println("  SD of naive Z-scores: ", std(zn, dims = 1))
    println("  SD of robust Z-scores: ", std(zs, dims = 1))
    println("  Proportion |Z|>2 based on naive covariance:")
    println("    ", mean(abs.(zn) .> 2, dims = 1))
    println("  Proportion |Z|>2 based on robust covariance:")
    println("    ", mean(abs.(zs) .> 2, dims = 1))
end

function main(mode)
    v = ["Not using", "Using"]
    for fixef in [false, true]
        for clust in [false, true]
            println(
                @sprintf(
                    "%s fixed effects, %s clustering:",
                    v[fixef ? 2 : 1],
                    lowercase(v[clust ? 2 : 1])
                )
            )
            run(fixef, clust, mode)
            println("")
        end
    end
end

println("Fixed effects and clustering variables are the same:")
main(1)

println("Fixed effects and clustering variables are different:")
main(2)
