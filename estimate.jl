using LinearAlgebra, StatsKit, Distributions, Random, Parameters
using Optim, LineSearches
include("common.jl")
using .BLPCommon # export: BLPParams, BLP_market_share

struct Estimate
    name::String
    estimate::Float64
    se::Float64
    pvalue::Float64
end

function Base.show(io::IO, ::MIME"text/plain", e::Vector{Estimate})
    println(io, "Name \t Estimate \t SE \t p-value")
    for e in e
        r(x) = round(x, digits=4)
        println(io, "$(e.name) \t $(r(e.estimate)) \t $(r(e.se)) \t $(r(e.pvalue))")
    end
end

function delta_iteration(data, mu, params, tol=1e-12)
    @unpack beta, sigma = params

    # initial guess of delta
    prodnum = size(data, 1)
    delta0 = zeros(prodnum)
    s_data = data.s

    marketid = data.marketid

    function contractmap(delta0, mu)
        s_pred = BLP_market_share(marketid, delta0, mu, params)

        delta1 = delta0 + log.(s_data) - log.(s_pred)
        return delta1
    end
    delta1 = contractmap(delta0, mu)
    iter = 1
    while max(abs.(delta1 - delta0)...) > tol
        if iter % 50 == 0
            # println("Iteration: $iter, diff: $(norm(delta1 - delta0))")
        end
        delta0 = delta1
        delta1 = contractmap(delta0, mu)
        iter += 1
    end

    return delta1
end


function GMM_obj(sigma; data, NP=300)
    params = BLPParams(
        beta=[0, 0, 0],
        sigma=sigma
    )

    X = Matrix(data[:, ["x1", "x2", "x3"]])
    mu, ksi = BLP_draw_mu(X, params.sigma; NP=NP)


    delta = delta_iteration(data, mu, params)
    beta = (X'X) \ (X'delta) # OLS beta given sigma

    partial_share_delta = BLPCommon.partial_share_delta(data.marketid, delta, mu, params)
    partial_share_sigma = BLPCommon.partial_share_sigma(data.marketid, delta, mu, X, ksi, params)
    partial_delta_sigma = -partial_share_delta \ partial_share_sigma

    m = delta - X * beta
    grad = vec(2 .* (delta - X * beta)' * (partial_delta_sigma - X * ((X'X) \ X') * partial_delta_sigma))

    return m'm, m, grad, delta, beta # objective function, moment conditions, and gradient
end

function main()
    data = CSV.read("simdata.csv", DataFrame)
    X = Matrix(data[:, ["x1", "x2", "x3"]])

    function GMM_obj_wrapper_optim!(F, G, sigma)
        print("sigma: $sigma; ")
        obj, m, grad_computed = GMM_obj(sigma; data=data)
        println("obj: $obj")
        if G != nothing
            G[:] = grad_computed
        end
        if F != nothing
            return obj
        end
    end

    res = Optim.optimize(
        Optim.only_fg!(GMM_obj_wrapper_optim!),
        [0, 0, 0],
        [5, 5, 5],
        [0.1, 0.1, 0.1],
        Fminbox(LBFGS(linesearch=LineSearches.BackTracking(order=3))),
        Optim.Options(
            x_abstol=1e-5,
            f_abstol=1e-5,
            outer_x_abstol=1e-5,
            outer_f_abstol=1e-5,
            iterations=50,
            show_trace=true,
        )
    )

    sigma_hat = res.minimizer
    obj, m, G, delta_hat, beta_hat = GMM_obj(sigma_hat; data=data, NP=1000)
    G = G'
    Ω = mean(m) * mean(m)'
    se_sigma = sqrt.(diag(pinv(G'G) * (G' * Ω * G) * pinv(G'G)))

    # heteroskedasticity-robust standard errors
    se_beta = sqrt.(diag((X'X) \ (X' * diagm(diag(m * m')) * X) / (X'X)))

    pvalue(pivotal) = cdf(Normal(0, 1), (1 - abs(pivotal)) * 2)

    return [
        Estimate("sigma1", sigma_hat[1], se_sigma[1], pvalue(sigma_hat[1] / se_sigma[1])),
        Estimate("sigma2", sigma_hat[2], se_sigma[2], pvalue(sigma_hat[2] / se_sigma[2])),
        Estimate("sigma3", sigma_hat[3], se_sigma[3], pvalue(sigma_hat[2] / se_sigma[3])),
        Estimate("beta1", beta_hat[1], se_beta[1], pvalue(beta_hat[1] / se_beta[1])),
        Estimate("beta2", beta_hat[2], se_beta[2], pvalue(beta_hat[2] / se_beta[2])),
        Estimate("beta3", beta_hat[3], se_beta[3], pvalue(beta_hat[3] / se_beta[3]))
    ]
end