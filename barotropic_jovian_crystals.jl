using Oceananigans
using Oceananigans.ImmersedBoundaries: GridFittedBoundary
using GLMakie
using Statistics

include("PolarPlaneCoriolis.jl")

using .PolarPlaneCoriolis: PolarPlane

grid = RectilinearGrid(size=(128, 128), x=(-π, π), y=(-π, π), topology=(Bounded, Bounded, Flat))

# Uncomment to mask out domain corners
#@inline circle(x, y, z) = sqrt(x^2 + y^2) > π
#grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(circle))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = WENO5(),
                            coriolis = PolarPlane(f₀=10, γ=2))


u, v, w = model.velocities

ϵ(x, y, z) = 2 * (rand() - 1)
ρ(x, y, z) = exp(-(x^2 + y^2))
uᵢ(x, y, z) = ϵ(x, y, z) * ρ(x, y, z)

set!(model, u=uᵢ, v=uᵢ)

u, v, w = model.velocities
u .-= mean(u)
v .-= mean(v)

simulation = Simulation(model, Δt=0.05, stop_time=20)

progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

u, v, w = model.velocities

ω = ∂x(v) - ∂y(u)
s = sqrt(u^2 + v^2)

# We pass these operations to an output writer below to calculate and output them during the simulation.
filename = "jovian_crystals.jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s),
                                                      schedule = TimeInterval(0.6),
                                                      filename = filename,
                                                      overwrite_existing = true)

run!(simulation)

ωt = FieldTimeSeries(filename, "ω")
st = FieldTimeSeries(filename, "s")

t = ωt.times
Nt = length(t)

fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
slider = Slider(fig[2, 1], range=1:Nt, startvalue=1)
n = slider.value
ωⁿ = @lift interior(ωt[$n], :, :, 1)
heatmap!(ax, ωⁿ)

display(fig)
