"""
    AGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
"""
struct AGARCH{p, q, T<:AbstractFloat} <: UnivariateVolatilitySpec{T}
    coefs::Vector{T}
    function AGARCH{p, q, T}(coefs::Vector{T}) where {p, q, T}
        length(coefs) == nparams(AGARCH{p, q})  || throw(NumParamError(nparams(AGARCH{p, q}), length(coefs)))
        new{p, q, T}(coefs)
    end
end

"""
    AGARCH{p, q}(coefs) -> UnivariateVolatilitySpec

Construct an AGARCH specification with the given parameters.

# Example:
```jldoctest
julia> AGARCH{1, 1}([0.1, .9, .04])
```
"""
AGARCH{p, q}(coefs::Vector{T}) where {p, q, T}  = AGARCH{p, q, T}(coefs)

@inline nparams(::Type{<:AGARCH{p, q}}) where {p, q} = p+3*q+1
@inline nparams(::Type{<:AGARCH{p, q}}, subset) where {p, q} = isempty(subset) ? 1 : sum(subset) + 1

@inline presample(::Type{<:AGARCH{p, q}}) where {p, q} = max(p, q)

Base.@propagate_inbounds @inline function update!(
    ht, lht, zt, at, ::Type{<:AGARCH{p, q}}, garchcoefs,
    current_horizon=1
    ) where {p, q}
mht = garchcoefs[1] # Volatility model intercept...
@muladd begin
    for i = 1:p
        mht = mht + garchcoefs[i+1]*ht[end-i+1]
    end
    for i = 1:q
        # α_1, α_2, α_3, ..., α_q, λ_1, λ_2, λ_3, ..., λ_q, ρ_1, ρ_2, ρ_3, ..., ρ_q
        α = garchcoefs[i+1+p]
        λ = garchcoefs[i+q+1+p]
        ρ = garchcoefs[i+2*q+1+p]
        mht = mht + α*(abs(zt[end-i+1]-λ) - ρ*(zt[end-i+1]-λ))*ht[end-i+1]
    end
end
push!(ht, mht^2)
push!(lht, (mht > 0) ? log(mht) : -mht)
return nothing
end

@inline function uncond(::Type{<:AGARCH{p, q}}, coefs::Vector{T}) where {p, q, T}
    den=one(T)
    for i in 1:p
        den -= coefs[i+1]
    end
    for i in 1:q
        α = coefs[i+1+p]
        λ = coefs[i+q+1+p]
        ρ = coefs[i+2*q+1+p]
        # Approximation under z_t ~ N(0,1)
        κ = λ * erf(λ/sqrt(2)) + sqrt(2/π) * exp(-0.5*λ^2)
        den -= α*κ
    end
    h0 = coefs[1]/den
end

function startingvals(::Type{<:AGARCH{p,q}}, data::Array{T}) where {p, q, T}
    x0 = zeros(T, p+3*q+1) .+ 1e-6
    P = zero(T)
    if p ≥ 1
        x0[2] = 0.8
        P += x0[2]
    end
    if q ≥ 1
        x[2+p] = 0.05
        P += sqrt(2/π)*x[2+p]
    end
    x0[1] = sum(abs.(data)) / length(data) * (1 - P)
    return x0
end

function startingvals(TT::Type{<:AGARCH}, data::Array{T} , subset::Tuple) where {T}
	x0 = zeros(T, p+q+1)
    x0 = zeros(T, p+3*q+1) .+ 1e-6
    P = zero(T)
    if p ≥ 1
        x0[2] = 0.8
        P += x0[2]
    end
    if q ≥ 1
        x[2+p] = 0.05
        P += sqrt(2/π)*x[2+p]
    end
	mask = subsetmask(TT, subset)
	x0long = zeros(T, length(mask))
	x0long[mask] .= x0
    return x0long
end

function constraints(::Type{<:AGARCH{p,q}}, ::Type{T}) where {p, q, T}
    # ω, β_1, ..., β_p, α_1, ..., α_q, λ_1, ..., λ_q, ρ_1, ..., ρ_q
    lower = zeros(T, p+3*q+1)
    lower[1] = 1e-6
    lower[(2+p+q):(1+p+2*q)] .= -10.0
    lower[(2+p+2*q):end] .= 1e-6 - 1.0

    upper = ones(T, p+3*q+1)
    upper[1] = T(Inf)
    upper[(2+p+q):(1+p+2*q)] .= 10.0
    upper[(2+p+2*q):end] .= 1.0 - 1e-6

    return lower, upper
end

function coefnames(::Type{<:AGARCH{p,q}}) where {p, q}
    names = Array{String, 1}(undef, p+3*q+1)
    names[1] = "ω"
    names[2:(p+1)] .= (i -> "β"*subscript(i)).([1:p...])
    names[(p+2):(p+q+1)] .= (i -> "α"*subscript(i)).([1:p...])
    names[(p+q+2):(1+p+2*q)] .= (i -> "λ"*subscript(i)).([1:p...])
    names[(2+p+2*q):end] .= (i -> "ρ"*subscript(i)).([1:p...])
    return names
end

@inline function subsetmask(VS_large::Union{Type{AGARCH{p, q}}, Type{AGARCH{p, q, T}}}, subs) where {p, q, T}
	ind = falses(nparams(VS_large))
	subset = zeros(Int, 2)
	subset[4-length(subs):end] .= subs
	ind[1] = true
	ps = subset[1]
	qs = subset[2]
	@assert ps <= p
	@assert qs <= q
    ind[2:(2+ps-1)] .= true
    ind[(2+p):(2+p+qs-1)] .= true
	ind
end

@inline function subsettuple(VS_large::Union{Type{AGARCH{p, q}}, Type{AGARCH{p, q, T}}}, subsetmask) where {p, q, T}
	ps = 0
	qs = 0
	@inbounds @simd ivdep for i = 2 : p + 1
		ps += subsetmask[i]
	end
	@inbounds @simd ivdep for i = p + 2 : p + q + 1
		qs += subsetmask[i]
	end
	(ps, qs)
end