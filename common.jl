module BLPCommon
using Parameters, Distributions, Tullio

export BLPParams, BLP_draw_mu, BLP_market_share

@with_kw struct BLPParams
    beta = [0, 0, 0] # mean coef
    sigma = [0, 0, 0] # consumer taste heterogeneity
end

function BLP_draw_mu(X, sigma; NP=1000)
    ksi = rand(Normal(0, 1), 3, NP)
    return X * (sigma .* ksi), ksi
end

# compute market shares
function BLP_market_share(marketid, delta, mu, params)
    @unpack beta, sigma = params

    u = delta .+ mu # total utility: TotalProdNum x NP

    s = zeros(size(delta, 1)) # market share vector
    for mkt in unique(marketid)
        u_m = u[marketid.==mkt, :] # subset of utility matrix for each market

        # choice prob of each consumer
        # outside option normalied to 1
        exp_u = exp.(u_m) ./ (1 .+ sum(exp.(u_m), dims=1))

        s_m = mean(exp_u, dims=2) # averaging choice probability of each consumer
        s[marketid.==mkt] = s_m
    end
    return s
end

function partial_share_delta(marketid, delta, mu, params)
    @unpack beta, sigma = params

    u = delta .+ mu # total utility: TotalProdNum x NP
    S = zeros(size(u))
    for mkt in unique(marketid)
        u_m = u[marketid.==mkt, :] # subset of utility matrix for each market

        # choice prob of each consumer
        # outside option normalied to 1
        S[marketid.==mkt, :] = exp.(u_m) ./ (1 .+ sum(exp.(u_m), dims=1))
    end

    @tullio par_share_delta[j, k] := -(S[j, :]' * (S[k, :])) / size(S, 2)
    @tullio par_share_delta[j, j] = (S[j, :]' * (1 .- S[j, :])) / size(S, 2)

    return par_share_delta
end

function partial_share_sigma(marketid, delta, mu, X, ksi, params)
    @unpack beta, sigma = params

    u = delta .+ mu # total utility: TotalProdNum x NP
    S = zeros(size(u))
    par_share_sigma = zeros(size(u, 1), size(X, 2), size(u, 2))
    for mkt in unique(marketid)
        u_m = u[marketid.==mkt, :] # subset of utility matrix for each market

        # choice prob of each consumer
        # outside option normalied to 1
        S[marketid.==mkt, :] = exp.(u_m) ./ (1 .+ sum(exp.(u_m), dims=1))
        Smkt = S[marketid.==mkt, :]
        Xmkt = X[marketid.==mkt, :]
        @tullio partial[j, l, i] := Smkt[j, i] * ksi[l, i] * (Xmkt[j, l] - (Smkt[jj, i] * Xmkt[jj, l]))
        par_share_sigma[marketid.==mkt, :, :] = partial
    end

    par_share_sigma = mean(par_share_sigma, dims=3)[:, :, 1]

    return par_share_sigma
end

end