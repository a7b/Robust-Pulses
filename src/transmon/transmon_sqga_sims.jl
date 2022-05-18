WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "transmon", "system.jl"))

using Altro
using HDF5
#using Colors
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization
using LaTeXStrings
using FFTW

# paths
const EXPERIMENT_META = "transmon"
const EXPERIMENT_NAME = "transmon_square"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const A_MAX = 2π * 19e-3
const A_MAX_LIN = 19
const CONTROL_COUNT = 2
const STATE_COUNT = 4
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO
const SAMPLE_COUNT = 2
const ASTATE_SIZE = ASTATE_SIZE_BASE
const ACONTROL_SIZE = CONTROL_COUNT
# const SAMPLE_COUNT = 4
# state indices
const STATE1_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)
const STATE2_IDX = SVector{HDIM_ISO}(STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO)
const STATE3_IDX = SVector{HDIM_ISO}(STATE2_IDX[end] + 1:STATE2_IDX[end] + HDIM_ISO)
const STATE4_IDX = SVector{HDIM_ISO}(STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_ISO)
#control indices
const CONTROLS_IDX = 1:CONTROL_COUNT

struct Model{DO} <: AbstractModel
end
@inline RD.state_dim(::Model{DO}) where {DO} = (
    ASTATE_SIZE_BASE + DO * SAMPLE_COUNT * HDIM_ISO
)
# @inline RD.state_dim(::Model{DO}) where {DO} = (
#     ASTATE_SIZE_BASE + (SAMPLE_COUNT - STATE_COUNT) * HDIM_ISO + DO * SAMPLE_COUNT * HDIM_ISO
# )
@inline RD.control_dim(::Model) = ACONTROL_SIZE

# dynamics
const NEGI2_H0_ISO = 2 * NEGI_TRANSMON_NUMBER_ISO

function RD.discrete_dynamics(::Type{RK3}, model::Model{DO}, astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real) where {DO}
    negi_h = (
        NEGI_H0_ISO
        + acontrol[CONTROLS_IDX[1]] * NEGI_H1R_ISO
        + acontrol[CONTROLS_IDX[2]] * NEGI_H1I_ISO
    )
    h_prop = exp(negi_h * dt)
    state1_ = astate[STATE1_IDX]
    state2_ = astate[STATE2_IDX]
    state3_ = astate[STATE3_IDX]
    state4_ = astate[STATE4_IDX]

    state1 =  h_prop * state1_
    state2 = h_prop * state2_
    state3 =  h_prop * state3_
    state4 = h_prop * state4_


    astate_ = [
        state1; state2; state3; state4;
    ]

    return astate_
end
function RD.discrete_dynamics(::Type{RK3}, model::Model{DO}, astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real, H::Matrix) where {DO}
    negi_h = (
        H
        + acontrol[CONTROLS_IDX[1]] * NEGI_H1R_ISO
        + acontrol[CONTROLS_IDX[2]] * NEGI_H1I_ISO
    )
    h_prop = exp(negi_h * dt)
    state1_ = astate[STATE1_IDX]
    state2_ = astate[STATE2_IDX]
    state3_ = astate[STATE3_IDX]
    state4_ = astate[STATE4_IDX]

    state1 =  h_prop * state1_
    state2 = h_prop * state2_
    state3 =  h_prop * state3_
    state4 = h_prop * state4_

    astate_ = [
        state1; state2; state3; state4;
    ]

    return astate_
end
@inline discrete_dynamics!(x_::AbstractVector, ::Type{Q}, model::AbstractModel, x::AbstractVector,
                           u::AbstractVector, t::Real, dt::Real, H::Matrix) where {Q} = (
                               x_ .= RD.discrete_dynamics(Q, model, x, u, t, dt, H)
)

function square_controls()
    #time in ns
    factor = 1
    time = 1e3*factor/(4*A_MAX_LIN)
    dt = time/100
    acontrols = repeat([[A_MAX/factor, 0.0]], 101)
    return (acontrols, time, dt)
end


function gaussian_controls()
    #0.9545
    MAX_DRIVE = A_MAX/3
    time_sigma = (π/2)/0.9545/(MAX_DRIVE * sqrt(2π))
    times = LinRange(-2*time_sigma, 2*time_sigma, 100)
    time = time_sigma*4
    dt = times[2] - times[1]
    contx = MAX_DRIVE * Base.exp.(-(times).^2/(2*time_sigma^2))
    acontrols = [[contx[k], 0.0] for k=1:length(contx)]
    #print(acontrols)
    append!(acontrols, [[0.0,0.0]])
    #print(acontrols)
    return (acontrols, time, dt)
end
function forward_pass_gaussian(derivative_order=0, integrator_type=rk3)
    (acontrols, evolution_time, dt) = gaussian_controls()

    knot_count = Integer(floor(evolution_time / dt)) + 1
    rdi = IT_RDI[integrator_type]
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    time = 0.
    astate = zeros(n)
    astate[STATE1_IDX] = IS1_ISO
    astate[STATE2_IDX] = IS2_ISO
    astate[STATE3_IDX] = IS3_ISO
    astate[STATE4_IDX] = IS4_ISO
    astate = SVector{n}(astate)

    for i = 1:knot_count - 1
        astate = SVector{n}(RD.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt))
        time = time + dt
    end

    res = Dict(
        "final state |g>" => get_vec_uniso(astate[STATE1_IDX]),
        "final state |e>" => get_vec_uniso(astate[STATE2_IDX]),
        "final state |g> + i|e>" => get_vec_uniso(astate[STATE3_IDX]),
        "final state |g> - |e>" => get_vec_uniso(astate[STATE4_IDX])
    )

    return res
end


function forward_pass(derivative_order=0, integrator_type=rk3)
    (acontrols, evolution_time, dt) = square_controls()

    knot_count = Integer(floor(evolution_time / dt)) + 1
    rdi = IT_RDI[integrator_type]
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    time = 0.
    astate = zeros(n)
    astate[STATE1_IDX] = IS1_ISO
    astate[STATE2_IDX] = IS2_ISO
    astate[STATE3_IDX] = IS3_ISO
    astate[STATE4_IDX] = IS4_ISO
    astate = SVector{n}(astate)

    for i = 1:knot_count - 1
        astate = SVector{n}(RD.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt))
        time = time + dt
    end

    res = Dict(
        "final state |g>" => get_vec_uniso(astate[STATE1_IDX]),
        "final state |e>" => get_vec_uniso(astate[STATE2_IDX]),
        "final state |g> + i|e>" => get_vec_uniso(astate[STATE3_IDX]),
        "final state |g> - |e>" => get_vec_uniso(astate[STATE4_IDX])
    )

    return res
end


function gen_dparam(derivative_order=0, trial_count=500, sigma_max=1e-4, save=true; gauss = true)
    # grab relevant information
    if gauss
        (U_, evolution_time, dt) = gaussian_controls()
    else
        (U_, evolution_time, dt) = square_controls()
    end

    integrator_type=rk3
    # set up problem
    n = size(NEGI_H0_ISO, 1)
    mh1 = zeros(n, n) .= NEGI_H0_ISO
    #Hs = [H for H in (mh1, NEGI_H1R_ISO, NEGI_H1I_ISO)]
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    N = Int(floor(evolution_time / dt)) + 1
    U = U_
    pop!(U)
    X = [zeros(n) for i = 1:N]
    ts = [dt * (k - 1) for k = 1:N]
    # initial state
    x0 = zeros(n)
    x0[STATE1_IDX] = IS1_ISO
    x0[STATE2_IDX] = IS2_ISO
    x0[STATE3_IDX] = IS3_ISO
    x0[STATE4_IDX] = IS4_ISO
    X[1] = x0
    # target state
    # xf = zeros(n)
    # cavity_state_ = cavity_state(target_level)
    # ψT = kron(cavity_state_, TRANSMON_G)
    # xf[model.state1_idx] = get_vec_iso(ψT)
    # xf = V(xf)
    xf = zeros(n)
    ψT = get_vec_uniso(XPI_G_ISO)
    ψT2 = get_vec_uniso(XPI_E_ISO)
    ψT3 = get_vec_uniso(XPI_3_ISO)
    ψT4 = get_vec_uniso(XPI_4_ISO)
    xf[STATE1_IDX] = XPI_G_ISO
    xf[STATE2_IDX] = XPI_E_ISO
    xf[STATE3_IDX] = XPI_3_ISO
    xf[STATE4_IDX] = XPI_4_ISO

    # generate parameters
    ωq_dev_max = 2π * 2e-3
    devs = Array(range(-ωq_dev_max, stop=ωq_dev_max, length=2 * trial_count + 1))
    fracs = map(d -> d/1, devs)
    negi_h0rot_iso(dev) = get_mat_iso(
        - 1im * (
            dev * TRANSMON_NUMBER  + α/2 * TRANSMON_QUAD
        )
    )
    negi_h0s = map(negi_h0rot_iso, devs)
    gate_errors = zeros(2 * trial_count + 1)
    gate_errors2 = zeros(2 * trial_count + 1)
    gate_errors3 = zeros(2 * trial_count + 1)
    gate_errors4 = zeros(2* trial_count + 1)
    # collect gate errors
    for (i, negi_h0) in enumerate(negi_h0s)
        mh1 .= negi_h0
        # rollout
        for k = 1:N-1
            discrete_dynamics!(X[k + 1], IT_RDI[integrator_type], model, X[k], U[k], ts[k], dt, mh1 )
        end
        ψN = get_vec_uniso(X[N][STATE1_IDX])
        ψn2 = get_vec_uniso(X[N][STATE2_IDX])
        ψn3 = get_vec_uniso(X[N][STATE3_IDX])
        ψn4 = get_vec_uniso(X[N][STATE4_IDX])

        gate_error = 1 - abs(ψT'ψN)^2
        gate_error2 = 1 - abs(ψT2'ψn2)^2
        gate_error3 = 1 - abs(ψT3'ψn3)^2
        gate_error4 = 1 - abs(ψT4'ψn4)^2
        gate_errors[i] = gate_error
        gate_errors2[i] = gate_error2
        gate_errors3[i] = gate_error3
        gate_errors4[i] = gate_error4
    end
    # save
    data_file_path = generate_file_path("h5", "dparams", SAVE_PATH)
    if save
        h5open(data_file_path, "w") do data_file
            write(data_file, "gate_errors", gate_errors)
            write(data_file, "gate_errors2", gate_errors2)
            write(data_file, "gate_errors3", gate_errors3)
            write(data_file, "gate_errors4", gate_errors4)
            write(data_file, "devs", devs)
            write(data_file, "fracs", fracs)
        end
    end

    return data_file_path
end




function read_population(save_file_path; title="", xlabel="Time (ns)", ylabel="Population",
                         legend=:bottomright)
    # grab
    save_file = read_save(save_file_path)
    transmon_state_count = save_file["transmon_state_count"]
    hdim_iso = save_file["hdim_iso"]
    ts = save_file["ts"]
    astates = save_file["astates"]
    N = size(astates, 1)
    d = Int(hdim_iso/2)
    state1_idx = Array(1:hdim_iso)
    state2_idx = Array(1 + hdim_iso: hdim_iso + hdim_iso)
    state3_idx = Array(1 + hdim_iso + hdim_iso: hdim_iso + hdim_iso + hdim_iso)
    state4_idx = Array(1 + hdim_iso + hdim_iso + hdim_iso: hdim_iso + hdim_iso + hdim_iso + hdim_iso)

    ψ = get_vec_uniso(astates[N, state1_idx])
    ψ2 = get_vec_uniso(astates[N, state2_idx])
    ψ3 = get_vec_uniso(astates[N, state3_idx])
    ψ4 = get_vec_uniso(astates[N, state4_idx])

    return [ψ, ψ2, ψ3, ψ4]
end
function plot_population(save_file_path; title="", xlabel="Time (ns)", ylabel="Population",
                         legend=:bottomright)
    # grab
    save_file = read_save(save_file_path)
    transmon_state_count = save_file["transmon_state_count"]
    hdim_iso = save_file["hdim_iso"]
    ts = save_file["ts"]
    astates = save_file["astates"]
    N = size(astates, 1)
    d = Int(hdim_iso/2)
    state1_idx = Array(1:hdim_iso)
    state2_idx = Array(1 + hdim_iso: hdim_iso + hdim_iso)
    state3_idx = Array(1 + hdim_iso + hdim_iso: hdim_iso + hdim_iso + hdim_iso)
    state4_idx = Array(1 + hdim_iso + hdim_iso + hdim_iso: hdim_iso + hdim_iso + hdim_iso + hdim_iso)
    # make labels
    transmon_labels = ["g", "e", "f", "h"][1:transmon_state_count]

    # plot
    fig = Plots.plot(dpi=DPI, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    pops = zeros(N, d)
    pops2 = zeros(N, d)
    pops3 = zeros(N, d)
    pops4 = zeros(N, d)
    for k = 1:N
        ψ = get_vec_uniso(astates[k, state1_idx])
        ψ2 = get_vec_uniso(astates[k, state2_idx])
        ψ3 = get_vec_uniso(astates[k, state3_idx])
        ψ4 = get_vec_uniso(astates[k, state4_idx])
        pops[k, :] = map(x -> abs(x)^2, ψ)
        pops2[k, :] = map(x -> abs(x)^2, ψ2)
        pops3[k, :] = map(x -> abs(x)^2, ψ3)
        pops4[k, :] = map(x -> abs(x)^2, ψ4)
    end
    styles = [:solid, :dash, :dot, :dashdot]
    for i = 1:d
        label = transmon_labels[i]
        style = styles[i]
        Plots.plot!(ts, pops[:, i], linestyle = style, lc = "cornflowerblue"  , label=label)
        Plots.plot!(ts, pops2[:, i], linestyle = style, lc = "darkorange", label = label)
        Plots.plot!(ts, pops3[:, i], linestyle = style, lc = "darkseagreen", label = label)
        Plots.plot!(ts, pops4[:, i], linestyle = style, lc = "violet", label = label)
    end
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end



function plot_dparam(data_file_paths; labels=["|g⟩", "|e⟩", "|g⟩ + i|e⟩", "|g⟩ - |e⟩"], legend=:bottomright)
    # grab
    gate_errors = []
    gate_errors2 = []
    gate_errors3 = []
    gate_errors4 = []

    fracs = []
    for data_file_path in data_file_paths
        (gate_errors_, gate_errors2_, gate_errors3_, gate_errors4_, fracs_) = h5open(data_file_path, "r") do data_file
            gate_errors_ = read(data_file, "gate_errors")
            gate_errors2_ = read(data_file, "gate_errors2")
            gate_errors3_ = read(data_file, "gate_errors3")
            gate_errors4_ = read(data_file, "gate_errors4")

            fracs_ = read(data_file, "fracs")
            return (gate_errors_, gate_errors2_, gate_errors3_, gate_errors4_,  fracs_)
        end
        push!(gate_errors, gate_errors_)
        push!(gate_errors2, gate_errors2_)
        push!(gate_errors3, gate_errors3_)
        push!(gate_errors4, gate_errors4_)
        push!(fracs, fracs_)
    end
    # initial plot
    ytick_vals = Array(-9:1:-1)
    ytick_labels = ["1e$(pow)" for pow in ytick_vals]
    yticks = (-9:1:-1, ytick_labels)
    fig = Plots.plot(dpi=DPI, title="Gate Error vs Detuning", legend=legend)
    Plots.xlabel!("\$\\Delta\\omega_q \\, \\, \\textrm{(MHz)}\$")
    Plots.ylabel!("Gate Error")
    gate_errors_ = gate_errors[1]
    gate_errors2_ = gate_errors2[1]
    gate_errors3_ = gate_errors3[1]
    gate_errors4_ = gate_errors4[1]
    fracs_ = fracs[1]
    trial_count = Int((length(fracs_) - 1)/2)
    gate_errors__ = zeros(trial_count + 1)
    gate_errors2__ = zeros(trial_count + 1)
    gate_errors3__ = zeros(trial_count + 1)
    gate_errors4__ = zeros(trial_count + 1)
    # average
    mid = trial_count + 1
    fracs__ = fracs_[mid:end]
    gate_errors__[1] = gate_errors_[mid]
    for j = 1:trial_count
        gate_errors__[j + 1] = (gate_errors_[mid - j] + gate_errors_[mid + j]) / 2
    end
    gate_errors2__[1] = gate_errors2_[mid]
    for j = 1:trial_count
        gate_errors2__[j + 1] = (gate_errors2_[mid - j] + gate_errors2_[mid + j]) / 2
    end
    gate_errors3__[1] = gate_errors3_[mid]
    for j = 1:trial_count
        gate_errors3__[j + 1] = (gate_errors3_[mid - j] + gate_errors3_[mid + j]) / 2
    end
    gate_errors4__[1] = gate_errors4_[mid]
    for j = 1:trial_count
        gate_errors4__[j + 1] = (gate_errors4_[mid - j] + gate_errors4_[mid + j]) / 2
    end
    #log_ge = map(x -> log10(x), gate_errors__)
    #label = isnothing(labels) ? nothing : labels[i]
    Plots.plot!(fracs__.*(1e3/2π), gate_errors__, label=labels[1])
    Plots.plot!(fracs__.*(1e3/2π), gate_errors2__, label=labels[2])
    Plots.plot!(fracs__.*(1e3/2π), gate_errors3__, label=labels[3])
    Plots.plot!(fracs__.*(1e3/2π), gate_errors4__, label=labels[4])
    plot_file_path = generate_file_path("png", "gate_error_plot", SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end

function sim_rob(gauss = true)
    data_path = gen_dparam(gauss = gauss)
    plot_dparam([data_path])
end

function create_square_pulse()
    time = 1e3/(2 * A_MAX_LIN)

end
