using Oceananigans
using Oceananigans.ImmersedBoundaries: GridFittedBoundary
using GLMakie
using Statistics

include("PolarPlaneCoriolis.jl")

using .PolarPlaneCoriolis: PolarPlane

Nh = 128
architecture = CPU()
f₀ = 3.52e-4
γ = 7.87e-20
U = 80
Lᵧ = (U / γ)^(1/3)
L = 6Lᵧ

@show τᵧ = Lᵧ / U

grid = RectilinearGrid(architecture;
                       size = (Nh, Nh),
                       x = (-L, L),
                       y = (-L, L),
                       topology = (Bounded, Bounded, Flat))

# Uncomment to mask out domain corners
@inline circle(x, y, z) = sqrt(x^2 + y^2) > L
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(circle))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = WENO5(),
                            coriolis = PolarPlane(; f₀, γ))


u, v, w = model.velocities

λ = L / 4
ϵ(x, y, z) = 2 * (rand() - 1)
ρ(x, y, z) = exp(-(x^2 + y^2) / 2λ^2)
uᵢ(x, y, z) = U * ϵ(x, y, z) * ρ(x, y, z)

set!(model, u=uᵢ, v=uᵢ)

u, v, w = model.velocities
u .-= mean(u)
v .-= mean(v)

Δt = 0.1 * grid.Δxᶜᵃᵃ / U
simulation = Simulation(model; Δt, stop_time=10τᵧ)

function progress(sim)
    msg = string("Iter: ", iteration(sim),
                 ", t: ", prettytime(sim))
    @info msg
    return nothing
end
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

u, v, w = model.velocities
ω = ∂x(v) - ∂y(u)

# We pass these operations to an output writer below to calculate and output them during the simulation.
filename = "low_res_jovian_crystals.jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω); filename,
                                                      schedule = IterationInterval(100),
                                                      overwrite_existing = true)

run!(simulation)

ωt = FieldTimeSeries(filename, "ω")

t = ωt.times
Nt = length(t)

@show ωlim = maximum(ωt[end]) / 2

fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value
ωⁿ = @lift interior(ωt[$n], :, :, 1)
heatmap!(ax, ωⁿ, colormap=:delta, colorrange=(-ωlim, ωlim))

display(fig)

record(fig, "low_res_barotropic_jovian_crystals.mp4", 1:Nt, framerate=24) do nn
    @info "Recording frame $nn of $Nt..."
    n[] = nn
end

