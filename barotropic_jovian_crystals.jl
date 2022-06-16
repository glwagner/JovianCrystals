using Oceananigans
using Oceananigans.ImmersedBoundaries: GridFittedBoundary
using GLMakie
using Statistics

include("GammaParaboloids.jl")

using .GammaParaboloids: GammaParaboloid

grid = RectilinearGrid(size=(128, 128), x=(-π, π), y=(-π, π), topology=(Bounded, Bounded, Flat))

# Uncomment to mask out domain corners
#@inline circle(x, y, z) = sqrt(x^2 + y^2) > π
#grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(circle))

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            advection = WENO5(),
                            #coriolis = BetaPlane(f₀=1, β=0.1))
                            coriolis = GammaParaboloid(f₀=1, γ=0.8))


u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

simulation = Simulation(model, Δt=0.1, stop_time=50)

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
ωⁿ = ωt[end]

fig = Figure()
ax = Axis(fig[1, 1], aspect=1)
heatmap!(ax, interior(ωⁿ, :, :, 1))
display(fig)
