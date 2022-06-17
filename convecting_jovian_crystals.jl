using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: GridFittedBoundary
using GLMakie
using Statistics

include("PolarPlaneCoriolis.jl")

using .PolarPlaneCoriolis: PolarPlane

Nh = 384
Nz = 128
architecture = GPU()
f₀ = 3.52e-4
γ = 7.87e-20
U = 80
Lᵧ = (U / γ)^(1/3)
L = 6Lᵧ
H = Lᵧ / 5

@show τᵧ = Lᵧ / U

# τᵇ = (H^2 / Qᵇ)^(1/3)
# -> Q = H^2 / τᵧ^3
Qᵇ = 1e6 * H^2 / τᵧ^3

grid = RectilinearGrid(architecture;
                       size = (Nh, Nh, Nz),
                       x = (-L, L),
                       y = (-L, L),
                       z = (0, H),
                       topology = (Bounded, Bounded, Bounded))

# Uncomment to mask out domain corners
@inline circle(x, y, z) = sqrt(x^2 + y^2) > 0.98L
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(circle))

b_top_bc = FluxBoundaryCondition(Qᵇ)
b_bottom_bc = FluxBoundaryCondition(Qᵇ)
b_bcs = FieldBoundaryConditions(top=b_top_bc, bottom=b_bottom_bc)

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
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
simulation = Simulation(model; Δt, stop_time=60days)

wall_time = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_time[])

    msg = string("Iteration: ", iteration(sim),
                 ", time: ", prettytime(sim),
                 ", Δt: ", prettytime(sim.Δt),
                 ", wall time: ", prettytime(elapsed))
    @info msg

    wall_time[] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

wizard = TimeStepWizard(cfl=0.5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

u, v, w = model.velocities
b = model.tracers.b

ω = ∂x(v) - ∂y(u)

# We pass these operations to an output writer below to calculate and output them during the simulation.
filename = "convecting_jovian_crystals"

k = round(Int, grid.Nz/2)

simulation.output_writers[:mid] = JLD2OutputWriter(model, (; ω, u, v, w, b),
                                                      schedule = IterationInterval(10),
                                                      filename = filename * "_mid.jld2",
                                                      indices = (:, :, k),
                                                      overwrite_existing = true)

simulation.output_writers[:bottom] = JLD2OutputWriter(model, (; ω, u, v, w, b),
                                                      schedule = IterationInterval(10),
                                                      filename = filename * "_bottom.jld2",
                                                      indices = (:, :, 8),
                                                      overwrite_existing = true)

l = round(Int, grid.Ny/2)
simulation.output_writers[:slice] = JLD2OutputWriter(model, (; ω, u, v, w, b),
                                                      schedule = IterationInterval(10),
                                                      filename = filename * "_slice.jld2",
                                                      indices = (:, l, :),
                                                      overwrite_existing = true)

run!(simulation)

#=
ωt = FieldTimeSeries(filename, "ω")
wt = FieldTimeSeries(filename, "w")

t = ωt.times
Nt = length(t)

fig = Figure()
ax1 = Axis(fig[1, 1], aspect=1)
ax2 = Axis(fig[1, 2], aspect=1)
slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
n = slider.value
ωⁿ = @lift interior(ωt[$n], :, :, 1)
wⁿ = @lift interior(wt[$n], :, :, 1)
heatmap!(ax1, ωⁿ)
heatmap!(ax2, wⁿ)

display(fig)
=#
