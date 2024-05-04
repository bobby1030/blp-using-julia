using LinearAlgebra, StatsKit, Distributions, Random

include("common.jl")
using .BLPCommon # export: BLPParams, BLP_market_share

Random.seed!(20240503)

### Define simulation constants ###

# number of products in each market
prodnum = [rand(Poisson(5)) for _ in 1:100] # 100 markets with mean of 5 products

# true parameters
params = BLPParams(
    beta=[0.8, 0.5, 0.3],
    sigma=[0.1, 0.2, 0.25]
)

### End of simulation constants ###

### Generate data ###
function simulate_data(
    prodnum, # a vector of numbers of product in each market
    params::BLPParams # true parameters
)
    MKT = length(prodnum)
    TotalProdNum = sum(prodnum)

    # product characteristics matrix: TotalProdNum x 3
    X = rand(Normal(0, 1), TotalProdNum, 3)
    marketid = vcat([repeat([mkt], prodnum[mkt]) for mkt in 1:MKT]...)

    delta = X * params.beta # product-specific utility: TotalProdNum x 1
    mu, ksi = BLP_draw_mu(X, params.sigma) # consumer-taste matrix: TotalProdNum x NP

    s = BLP_market_share(marketid, delta, mu, params)

    df = DataFrame(marketid=marketid, s=s)
    dfX = DataFrame(X, :auto)
    df = hcat(df, dfX)

    return df
end

df = simulate_data(prodnum, params)
CSV.write("simdata.csv", df)