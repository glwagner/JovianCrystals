module GammaParaboloids

using Oceananigans.Grids: Face, Center, xnode, ynode
using Oceananigans.Operators: Δx_qᶜᶠᶜ, Δy_qᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, ℑyᵃᶜᵃ, ℑxᶜᵃᵃ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Advection: EnergyConservingScheme, EnstrophyConservingScheme
using Oceananigans.Coriolis: AbstractRotation

using Printf

import Oceananigans.Coriolis: x_f_cross_U, y_f_cross_U, z_f_cross_U 

struct GammaParaboloid{S, FT} <: AbstractRotation
    f₀ :: FT
    γ :: FT
    scheme :: S
end

"""
    GammaParaboloid([FT=Float64;] rotation_rate=Ω_Earth, scheme=EnergyConservingScheme()))

Returns a parameter object for a "tangent paraboloid" approximation to Coriolis forces near the
pole of a rotating sphere.
"""
GammaParaboloid(FT::DataType=Float64; f₀, γ, scheme::S=EnergyConservingScheme()) where S =
    GammaParaboloid{S, FT}(f₀, γ, scheme)

@inline function fᶠᶠᵃ(i, j, k, grid, coriolis::GammaParaboloid)
    x = xnode(Face(), Face(), Center(), i, j, k, grid)
    y = ynode(Face(), Face(), Center(), i, j, k, grid)
    f₀ = coriolis.f₀
    γ = coriolis.γ
    f = f₀ - γ / 2 * (x^2 + y^2)
    return f
end

@inline z_f_cross_U(i, j, k, grid, coriolis::GammaParaboloid, U) = zero(grid)

#####
##### Enstrophy-conserving scheme
#####

const CoriolisEnstrophyConserving = GammaParaboloid{<:EnstrophyConservingScheme}

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnstrophyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, fᶠᶠᵃ, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Energy-conserving scheme
#####

const CoriolisEnergyConserving = GammaParaboloid{<:EnergyConservingScheme}

@inline f_ℑx_vᶠᶠᵃ(i, j, k, grid, coriolis, v) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑxᶠᵃᵃ(i, j, k, grid, Δx_qᶜᶠᶜ, v)
@inline f_ℑy_uᶠᶠᵃ(i, j, k, grid, coriolis, u) = fᶠᶠᵃ(i, j, k, grid, coriolis) * ℑyᵃᶠᵃ(i, j, k, grid, Δy_qᶠᶜᶜ, u)

@inline x_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds - ℑyᵃᶜᵃ(i, j, k, grid, f_ℑx_vᶠᶠᵃ, coriolis, U[2]) / Δxᶠᶜᶜ(i, j, k, grid)

@inline y_f_cross_U(i, j, k, grid, coriolis::CoriolisEnergyConserving, U) =
    @inbounds + ℑxᶜᵃᵃ(i, j, k, grid, f_ℑy_uᶠᶠᵃ, coriolis, U[1]) / Δyᶜᶠᶜ(i, j, k, grid)

#####
##### Show
#####

function Base.show(io::IO, c::GammaParaboloid{FT}) where FT
    name = "GammaParaboloid{$FT}"
    str = @sprintf("%s: f₀ = %.2e, γ = %.2e)", name, c.f₀, c.γ)
    return print(io, msg)
end

end # module
