using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: GridFittedBoundary
using GLMakie
using Statistics

include("PolarPlaneCoriolis.jl")

using .PolarPlaneCoriolis: PolarPlane

Nh = 256
Nz = 64
architecture = GPU()
f₀ = 3.52e-4
γ = 7.87e-20
U = 80
Lᵧ = (U / γ)^(1/3)
L = 6Lᵧ
H = Lᵧ / 5
τᵧ = Lᵧ / U

# τᵇ = (H^2 / Qᵇ)^(1/3)
# -> Q = R^3 * H^2 / τᵧ^3
# where R = τᵇ / τᵧ
Qᵇ = 1e-4 #1e-1 * H^2 / τᵧ^3

w★ = (Qᵇ * H)^(1/3)
@show Ro★ = w★ / (f₀ * H)

grid = RectilinearGrid(architecture;
                       size = (Nh, Nh, Nz),
                       x = (-L, L),
                       y = (-L, L),
                       z = (0, H),
                       topology = (Bounded, Bounded, Bounded))

# Uncomment to mask out domain corners
@inline circle(x, y, z) = sqrt(x^2 + y^2) > 0.9L
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(circle))

b_top_bc = FluxBoundaryCondition(Qᵇ)
b_bottom_bc = FluxBoundaryCondition(Qᵇ)
b_bcs = FieldBoundaryConditions(top=b_top_bc, bottom=b_bottom_bc)

model = NonhydrostaticModel(; grid,
                            timestepper = :RungeKutta3,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            advection = WENO5(),
                            closure = AnisotropicMinimumDissipation(),
                            boundary_conditions = (; b=b_bcs),
                            coriolis = PolarPlane(; f₀, γ))

u, v, w = model.velocities

wΔ = (Qᵇ * grid.Δzᵃᵃᶜ)^(1/3)

λ = L / 4
ρ(x, y, z) = exp(-(x^2 + y^2) / 2λ^2)
ϵ(x, y, z) = 2 * (rand() - 1)
uᵢ(x, y, z) = U * ϵ(x, y, z) * ρ(x, y, z)
wᵢ(x, y, z) = 1e-3 * wΔ * ϵ(x, y, z)

set!(model, u=uᵢ, v=uᵢ, w=wᵢ)

u, v, w = model.velocities
u .-= mean(u)
v .-= mean(v)

# Δt = 0.1 * grid.Δxᶜᵃᵃ / U
Δt = 0.1 * grid.Δzᵃᵃᶜ / wΔ
simulation = Simulation(model; Δt, stop_iteration=2000) #stop_time=10days)

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

wizard = TimeStepWizard(cfl=0.3)
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

ωt = FieldTimeSeries(filename * "_mid.jld2", "ω")
wt = FieldTimeSeries(filename * "_mid.jld2", "w")

wxzt = FieldTimeSeries(filename * "_slice.jld2", "w")
bxzt = FieldTimeSeries(filename * "_slice.jld2", "b")

wlim = maximum(interior(wt[end], :, :, 1)) / 2
ωlim = maximum(interior(ωt[end], :, :, 1)) / 2
bmin = minimum(interior(bxzt[end], :, 1, :))
bmax = maximum(interior(bxzt[end], :, 1, :))

t = ωt.times
Nt = length(t)

fig = Figure(resolution=(1600, 1200))
ax1 = Axis(fig[1, 1], aspect=1)
ax2 = Axis(fig[1, 2], aspect=1)

ax3 = Axis(fig[2, 1], aspect=1)
ax4 = Axis(fig[2, 2], aspect=1)

slider = Slider(fig[3, 1:3], range=1:Nt, startvalue=1)
n = slider.value

ωⁿ = @lift interior(ωt[$n], :, :, 1)
wⁿ = @lift interior(wt[$n], :, :, 1)

wxzⁿ = @lift interior(wxzt[$n], :, 1, :)
bxzⁿ = @lift interior(bxzt[$n], :, 1, :)

heatmap!(ax1, ωⁿ, colormap=:delta, colorrange=(-ωlim, ωlim))
heatmap!(ax2, wⁿ, colormap=:delta, colorrange=(-wlim, wlim))

heatmap!(ax3, wxzⁿ, colormap=:delta, colorrange=(-wlim, wlim))
heatmap!(ax4, bxzⁿ, colorrange=(bmin, bmax))

display(fig)
