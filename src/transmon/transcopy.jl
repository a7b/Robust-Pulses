"""
transmon.jl - vanilla
"""

# paths
WDIR = abspath(@__DIR__, "../../")
const EXPERIMENT_META = "spin"
include(joinpath(WDIR, "src", EXPERIMENT_META, EXPERIMENT_META * ".jl"))
const EXPERIMENT_NAME = "transmon"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_NAME, "robust"))

using Altro
using HDF5
using LinearAlgebra
using StaticArrays

# redefine constants
#const ω_q = 2π * 4.96 #GHz
#anharmonicity
#const α = -2π * 0.143  #GHz
const A_MAX = 2π * 33e-3


# model
struct Model{TH,Tis,Tic} <: AbstractModel
    # problem size
    n::Int
    m::Int
    control_count::Int
    # problem
    Hs::Vector{TH}
    derivative_count::Int
    # indices
    state1_idx::Tis
    controls_idx::Tic
    dcontrols_idx::Tic
    dstate1_idx::Tis
    d2controls_idx::Tic
end

function Model(M_, Md_, V_, Hs, derivative_count)
    # problem size
    control_count = 2
    state_count = 1 + derivative_count
    n = state_count * HDIM_ISO + 2 * control_count
    m = control_count
    # state indices
    state1_idx = V(1:HDIM_ISO)
    controls_idx = V(state1_idx[end] + 1:state1_idx[end] + control_count)
    dcontrols_idx = V(controls_idx[end] + 1:controls_idx[end] + control_count)
    dstate1_idx = V(dcontrols_idx[end] + 1:dcontrols_idx[end] + HDIM_ISO)

    # control indices
    d2controls_idx = V(1:control_count)
    # types
    TH = typeof(Hs[1])
    Tis = typeof(state1_idx)
    Tic = typeof(controls_idx)
    return Model{TH,Tis,Tic}(n, m, control_count, Hs, derivative_count, state1_idx,
                             controls_idx, dcontrols_idx, dstate1_idx, d2controls_idx)
end

@inline Base.size(model::Model) = model.n, model.m
# vector and matrix constructors (use CPU arrays)
@inline M(mat_) = mat_
@inline Md(mat_) = mat_
@inline V(vec_) = vec_

# dynamics
abstract type EXP <: Explicit end

function Altro.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real)
    # get hamiltonian and unitary
    H = dt * (
        model.Hs[1]
        + astate[model.controls_idx[1]] * model.Hs[2]
        + astate[model.controls_idx[2]] * model.Hs[3]
    )
    U = exp(H)
    # propagate state
    state1 = U * astate[model.state1_idx]
    # propagate controls
    controls = astate[model.dcontrols_idx] .* dt + astate[model.controls_idx]
    # propagate dcontrols
    dcontrols = acontrol[model.d2controls_idx] .* dt + astate[model.dcontrols_idx]
    # construct astate
    astate_ = [state1; controls; dcontrols]
    if (model.derivative_count >=1)
        dstate1_ = astate[model.dstate1_idx]
        dstate1 = U * (dstate1_ + dt * NEGI_TRANSMON_NUMBER_ISO * astate[model.state1_idx])
        astate_ = [astate_; dstate1]
    end
    return astate_
end

# main
function run_traj(;evolution_time=10., dt=1e-2, verbose=true,
                  derivative_count=0, smoke_test=false, save=true, benchmark=false,
                  pn_steps=2, max_penalty=1e11,
                  max_iterations=Int64(2e5),
                  max_cost=1e8, ilqr_ctol=1e-4, ilqr_gtol=1e-4,
                  ilqr_max_iterations=300, max_state_value=1e10,
                  max_control_value=1e10, qs=[1e0, 1e0, 1e0, 5e-2, 5e-1], pn=true,
                  nf = false, nf_tol = 0., control_path = "C:\\Users\\a7b\\qoc-experiments\\out\\transmon\\controls13", al_outer = 60)
    # model configuration
    Hs = [M(H) for H in (NEGI_H0_ISO, NEGI_H1R_ISO, NEGI_H1I_ISO, NEGI_TRANSMON_NUMBER_ISO)]
    model = Model(M, Md, V, Hs, derivative_count)
    n, m = size(model)
    N = Int(floor(evolution_time / dt)) + 1
    t0 = 0.

    # initial state
    x0 = zeros(n)
    x0[model.state1_idx] = IS1_ISO
    x0 = V(x0)

    # final state

    xf = zeros(n)
    xf[model.state1_idx] = XPIBY2_ISO
    xf = V(xf)

    # bound constraints
    x_max_amid = fill(Inf, n)
    x_max_abnd = fill(Inf, n)
    x_min_amid = fill(-Inf, n)
    x_min_abnd = fill(-Inf, n)
    u_max_amid = fill(Inf, m)
    u_max_abnd = fill(Inf, m)
    u_min_amid = fill(-Inf, m)
    u_min_abnd = fill(-Inf, m)
    # constrain the control amplitudes
    x_max_amid[model.controls_idx] .= A_MAX
    x_min_amid[model.controls_idx] .= -A_MAX
    # control amplitudes go to zero at boundary
    x_max_abnd[model.controls_idx] .= 0
    x_min_abnd[model.controls_idx] .= 0
    # vectorize
    x_max_amid = V(x_max_amid)
    x_max_abnd = V(x_max_abnd)
    x_min_amid = V(x_min_amid)
    x_min_abnd = V(x_min_abnd)
    u_max_amid = V(u_max_amid)
    u_max_abnd = V(u_max_abnd)
    u_min_amid = V(u_min_amid)
    u_min_abnd = V(u_min_abnd)

    # constraints
    constraints = Altro.ConstraintList(n, m, N, M, V)
    bc_amid = BoundConstraint( x_max_amid, x_min_amid, u_max_amid, u_min_amid, n, m, M, V)
    add_constraint!(constraints, bc_amid, V(2:N-2))

    if nf
        nf_sense = nf_tol == 0. ? EQUALITY : INEQUALITY
        nf_nopop = NormConstraint(nf_sense, STATE, [3, 3 + HDIM], nf_tol, n, m, M, V)
        add_constraint!(constraints, nf_nopop, V(2:N-1))
    end
    gc_bnd_idxs = model.controls_idx
    gc_bnd_x = V(zeros(n))
    gc_bnd = GoalConstraint(gc_bnd_x, gc_bnd_idxs, n, m, M, V)
    add_constraint!(constraints, gc_bnd, V(N-1:N-1))
    goal_idxs = model.state1_idx
    gc_f = GoalConstraint(xf, goal_idxs, n, m, M, V)
    add_constraint!(constraints, gc_f, V(N:N))

    # initial trajectory
    X0 = [V(zeros(n)) for k = 1:N]
    X0[1] .= x0
    U0 = [V([
        fill(1e-6, model.control_count);
    ]) for k = 1:N-1]
    ts = V(zeros(N))
    ts[1] = t0
    for k = 1:N-1
        ts[k + 1] = ts[k] + dt
        Altro.discrete_dynamics!(X0[k + 1], EXP, model, X0[k], U0[k], ts[k], dt)
    end

    # cost function
    Q = V(zeros(n))
    Q[model.state1_idx] .= qs[1]
    Q[model.controls_idx] .= qs[2]
    Q[model.dcontrols_idx] .= qs[3]
    if model.derivative_count == 1
        Q[model.dstate1_idx] .= qs[4]
    end
    Q = Diagonal(Q)
    Qf = Q * N
    R = V(zeros(m))
    R[model.d2controls_idx] .= qs[5]
    R = Diagonal(R)
    objective = LQRObjective(Q, Qf, R, xf, n, m, N, M, V)

    # build problem
    prob = Problem(EXP, model, objective, constraints, X0, U0, ts, N, M, Md, V)
    # options
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    ilqr_max_iterations = smoke_test ? 1 : ilqr_max_iterations
    al_max_iterations = smoke_test ? 1 : al_outer
    n_steps = smoke_test ? 1 : pn_steps
    opts = SolverOptions(
        penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
        projected_newton=pn, ilqr_max_iterations=ilqr_max_iterations,
        al_max_iterations=al_max_iterations,
        iterations=max_iterations,
        max_cost_value=max_cost, ilqr_ctol=ilqr_ctol, ilqr_gtol=ilqr_gtol,
        max_state_value=max_state_value,
        max_control_value=max_control_value, al_vtol=1e-4
    )

    # solve
    solver = ALTROSolver(prob, opts)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end
    println("status: $(solver.stats.status)")

    # post-process
    acontrols_raw = Altro.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = Altro.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    state1_idx_arr = Array(model.state1_idx)
    controls_idx_arr = Array(model.controls_idx)
    dcontrols_idx_arr = Array(model.dcontrols_idx)
    d2controls_idx_arr = Array(model.d2controls_idx)
    max_v, max_v_info = Altro.max_violation_info(solver)
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "astates" => astates_arr,
        "dt" => dt,
        "ts" => ts,
        "state1_idx" => state1_idx_arr,
        "controls_idx" => controls_idx_arr,
        "dcontrols_idx" => dcontrols_idx_arr,
        "d2controls_idx" => d2controls_idx_arr,
        "evolution_time" => evolution_time,
        "max_v" => max_v,
        "max_v_info" => max_v_info,
        "qs" => qs,
        "iterations" => iterations_,
        "hdim_iso" => HDIM_ISO,
        "save_type" => Int(jl),
        "max_penalty" => max_penalty,
        "ilqr_ctol" => ilqr_ctol,
        "ilqr_gtol" => ilqr_gtol,
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost" => max_cost,
        "derivative_count" => derivative_count,
        "transmon_state_count" => TRANSMON_STATE_COUNT
    )

    # save
    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            for key in keys(result)
                write(save_file, key, result[key])
            end
        end
        result["save_file_path"] = save_file_path
    end

    result = benchmark ? benchmark_result : result
    plot_population(save_file_path)
    plot_controls([save_file_path], control_path)
    data_path = gen_dparam(save_file_path)
    plot_dparam([data_path])

    return result
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
    # make labels
    transmon_labels = ["g", "e", "f", "h"][1:transmon_state_count]

    # plot
    fig = Plots.plot(dpi=DPI, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)
    plot_file_path = generate_file_path("png", EXPERIMENT_NAME, SAVE_PATH)
    pops = zeros(N, d)
    for k = 1:N
        ψ = get_vec_uniso(astates[k, state1_idx])
        pops[k, :] = map(x -> abs(x)^2, ψ)
    end
    for i = 1:d
        label = transmon_labels[i]
        Plots.plot!(ts, pops[:, i], label=label)
    end
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end

function gen_dparam(save_file_path; trial_count=500, sigma_max=1e-4, save=true)
    # grab relevant information
    (evolution_time, dt, derivative_count, U_,
     ) = h5open(save_file_path, "r") do save_file
        evolution_time = read(save_file, "evolution_time")
        dt = read(save_file, "dt")
        derivative_count = read(save_file, "derivative_count")
        U_ = read(save_file, "acontrols")
        return (evolution_time, dt, derivative_count, U_)
    end
    # set up problem
    n = size(NEGI_H0_ISO, 1)
    mh1 = zeros(n, n) .= NEGI_H0_ISO
    Hs = [M(H) for H in (mh1, NEGI_H1R_ISO, NEGI_H1I_ISO)]
    model = Model(M, Md, V, Hs, derivative_count)
    n, m = size(model)
    N = Int(floor(evolution_time / dt)) + 1
    U = [U_[k, :] for k = 1:N-1]
    X = [zeros(n) for i = 1:N]
    ts = [dt * (k - 1) for k = 1:N]
    # initial state
    x0 = zeros(n)
    x0[model.state1_idx] = IS1_ISO
    x0 = V(x0)
    X[1] = x0
    # target state
    # xf = zeros(n)
    # cavity_state_ = cavity_state(target_level)
    # ψT = kron(cavity_state_, TRANSMON_G)
    # xf[model.state1_idx] = get_vec_iso(ψT)
    # xf = V(xf)
    xf = zeros(n)
    ψT = XPIBY2_subspace[:,1]
    xf[model.state1_idx] = XPIBY2_ISO
    xf = V(xf)
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
    # collect gate errors
    for (i, negi_h0) in enumerate(negi_h0s)
        mh1 .= negi_h0
        # rollout
        for k = 1:N-1
            Altro.discrete_dynamics!(X[k + 1], EXP, model, X[k], U[k], ts[k], dt)
        end
        ψN = get_vec_uniso(X[N][model.state1_idx])
        gate_error = 1 - abs(ψT'ψN)^2
        gate_errors[i] = gate_error
    end
    # save
    data_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
    if save
        h5open(data_file_path, "w") do data_file
            write(data_file, "save_file_path", save_file_path)
            write(data_file, "gate_errors", gate_errors)
            write(data_file, "devs", devs)
            write(data_file, "fracs", fracs)
        end
    end

    return data_file_path
end


function plot_dparam(data_file_paths; labels=nothing, legend=nothing)
    # grab
    gate_errors = []
    fracs = []
    for data_file_path in data_file_paths
        (gate_errors_, fracs_) = h5open(data_file_path, "r") do data_file
            gate_errors_ = read(data_file, "gate_errors")
            fracs_ = read(data_file, "fracs")
            return (gate_errors_, fracs_)
        end
        push!(gate_errors, gate_errors_)
        push!(fracs, fracs_)
    end
    # initial plot
    ytick_vals = Array(-9:1:-1)
    ytick_labels = ["1e$(pow)" for pow in ytick_vals]
    yticks = (-9:1:-1, ytick_labels)
    fig = Plots.plot(dpi=DPI, title=nothing, legend=legend)
    Plots.xlabel!("δω_q (MHz)")
    Plots.ylabel!("Gate Error")
    gate_errors_ = gate_errors[1]
    fracs_ = fracs[1]
    trial_count = Int((length(fracs_) - 1)/2)
    gate_errors__ = zeros(trial_count + 1)
    # average
    mid = trial_count + 1
    fracs__ = fracs_[mid:end]
    gate_errors__[1] = gate_errors_[mid]
    for j = 1:trial_count
        gate_errors__[j + 1] = (gate_errors_[mid - j] + gate_errors_[mid + j]) / 2
    end
    log_ge = map(x -> log10(x), gate_errors__)
    label = isnothing(labels) ? nothing : labels[i]
    Plots.plot!(fracs__.*(1e3/2π), gate_errors__, label=label)
    plot_file_path = generate_file_path("png", "gate_error_plot", SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end
