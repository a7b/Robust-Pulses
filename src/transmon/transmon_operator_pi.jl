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

# paths
const EXPERIMENT_META = "transmon"
const EXPERIMENT_NAME = "transmon_operator_pi"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const A_MAX = 2π * 6e-3
const CONTROL_COUNT = 2
const STATE_COUNT = 4
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 2 * CONTROL_COUNT
const SAMPLE_COUNT = 4
const ASTATE_SIZE = ASTATE_SIZE_BASE
const ACONTROL_SIZE = CONTROL_COUNT
# const SAMPLE_COUNT = 4
# state indices
const STATE1_IDX = SVector{HDIM_ISO}(1:HDIM_ISO)
const STATE2_IDX = SVector{HDIM_ISO}(STATE1_IDX[end] + 1:STATE1_IDX[end] + HDIM_ISO)
const STATE3_IDX = SVector{HDIM_ISO}(STATE2_IDX[end] + 1:STATE2_IDX[end] + HDIM_ISO)
const STATE4_IDX = SVector{HDIM_ISO}(STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_ISO)

const CONTROLS_IDX = STATE4_IDX[end] + 1:STATE4_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const DSTATE1_IDX = SVector{HDIM_ISO}(DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO)
const DSTATE2_IDX = SVector{HDIM_ISO}(DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + HDIM_ISO)
const DSTATE3_IDX = SVector{HDIM_ISO}(DSTATE2_IDX[end] + 1:DSTATE2_IDX[end] + HDIM_ISO)
const DSTATE4_IDX = SVector{HDIM_ISO}(DSTATE3_IDX[end] + 1:DSTATE3_IDX[end] + HDIM_ISO)

const D2STATE1_IDX = SVector{HDIM_ISO}(DSTATE2_IDX[end] + 1:DSTATE2_IDX[end] + HDIM_ISO)
const D2STATE2_IDX = SVector{HDIM_ISO}(D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + HDIM_ISO)
const D2STATE3_IDX = SVector{HDIM_ISO}(D2STATE2_IDX[end] + 1:D2STATE2_IDX[end] + HDIM_ISO)
const D2STATE4_IDX = SVector{HDIM_ISO}(D2STATE3_IDX[end] + 1:D2STATE3_IDX[end] + HDIM_ISO)

# const STATE3_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
# const STATE4_IDX = STATE3_IDX[end] + 1:STATE3_IDX[end] + HDIM_ISO
# const DSTATE1_IDX = STATE4_IDX[end] + 1:STATE4_IDX[end] + HDIM_ISO
# const DSTATE2_IDX = DSTATE1_IDX[end] + 1:DSTATE1_IDX[end] + HDIM_ISO
# const DSTATE3_IDX = DSTATE2_IDX[end] + 1:DSTATE2_IDX[end] + HDIM_ISO
# const DSTATE4_IDX = DSTATE3_IDX[end] + 1:DSTATE3_IDX[end] + HDIM_ISO
# const D2STATE1_IDX = DSTATE4_IDX[end] + 1:DSTATE4_IDX[end] + HDIM_ISO
# const D2STATE2_IDX = D2STATE1_IDX[end] + 1:D2STATE1_IDX[end] + HDIM_ISO
# const D2STATE3_IDX = D2STATE2_IDX[end] + 1:D2STATE2_IDX[end] + HDIM_ISO
# const D2STATE4_IDX = D2STATE3_IDX[end] + 1:D2STATE3_IDX[end] + HDIM_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
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
function RD.discrete_dynamics(::Type{RK3}, model::Model{DO}, astate::SVector,
                              acontrol::SVector, time::Real, dt::Real) where {DO}
    negi_h = (
        NEGI_H0_ISO
        + astate[CONTROLS_IDX[1]] * NEGI_H1R_ISO
        + astate[CONTROLS_IDX[2]] * NEGI_H1I_ISO
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

    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] .* dt
    dcontrols = astate[DCONTROLS_IDX] + acontrol[D2CONTROLS_IDX] .* dt

    astate_ = [
        state1; state2; state3; state4; controls; dcontrols;
    ]

    if DO >= 1
        dstate1_ = astate[DSTATE1_IDX]
        dstate1 = h_prop * (dstate1_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state1_)
        append!(astate_, dstate1)

        dstate2_ = astate[DSTATE2_IDX]
        dstate2 = h_prop * (dstate2_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state2_)
        append!(astate_, dstate2)

        dstate3_ = astate[DSTATE3_IDX]
        dstate3 = h_prop * (dstate3_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state3_)
        append!(astate_, dstate3)

        dstate4_ = astate[DSTATE4_IDX]
        dstate4 = h_prop * (dstate4_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state4_)
        append!(astate_, dstate4)
        # state3_ = astate[STATE3_IDX]
        # state3 = h_prop * state3_
        # state4_ = astate[STATE4_IDX]
        # state4 = h_prop * state4_
        # dstate1_ = astate[DSTATE1_IDX]
        # dstate1 = h_prop * (dstate1_ + dt * NEGI_H0_ISO * state1_)
        # dstate2_ = astate[DSTATE2_IDX]
        # dstate2 = h_prop * (dstate2_ + dt * NEGI_H0_ISO * state2_)
        # dstate3_ = astate[DSTATE3_IDX]
        # dstate3 = h_prop * (dstate3_ + dt * NEGI_H0_ISO * state3_)
        # dstate4_ = astate[DSTATE4_IDX]
        # dstate4 = h_prop * (dstate4_ + dt * NEGI_H0_ISO * state4_)
        # append!(astate_, [state3; state4; dstate1; dstate2; dstate3; dstate4])
    end
    if DO >= 2
        d2state1_ = astate[D2STATE1_IDX]
        d2state1 = h_prop * (d2state1_ + dt * NEGI2_H0_ISO * dstate1_)
        append!(astate_, d2state1)

        d2state2_ = astate[D2STATE2_IDX]
        d2state2 = h_prop * (d2state2_ + dt * NEGI2_H0_ISO * dstate2_)
        append!(astate_, d2state2)

        d2state3_ = astate[D2STATE3_IDX]
        d2state3 = h_prop * (d2state3_ + dt * NEGI2_H0_ISO * dstate3_)
        append!(astate_, d2state3)

        d2state4_ = astate[D2STATE4_IDX]
        d2state4 = h_prop * (d2state4_ + dt * NEGI2_H0_ISO * dstate4_)
        append!(astate_, d2state4)
        # d2state1_ = astate[D2STATE1_IDX]
        # d2state1 = h_prop * (d2state1_ + dt * NEGI2_H0_ISO * dstate1_)
        # d2state2_ = astate[D2STATE2_IDX]
        # d2state2 = h_prop * (d2state2_ + dt * NEGI2_H0_ISO * dstate2_)
        # d2state3_ = astate[D2STATE3_IDX]
        # d2state3 = h_prop * (d2state3_ + dt * NEGI2_H0_ISO * dstate3_)
        # d2state4_ = astate[D2STATE4_IDX]
        # d2state4 = h_prop * (d2state4_ + dt * NEGI2_H0_ISO * dstate4_)
        # append!(astate_, [d2state1; d2state2; d2state3; d2state4])
    end

    return astate_
end
function RD.discrete_dynamics(::Type{RK3}, model::Model{DO}, astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real, H::Matrix) where {DO}
    negi_h = (
        H
        + astate[CONTROLS_IDX[1]] * NEGI_H1R_ISO
        + astate[CONTROLS_IDX[2]] * NEGI_H1I_ISO
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
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] .* dt
    dcontrols = astate[DCONTROLS_IDX] + acontrol[D2CONTROLS_IDX] .* dt

    astate_ = [
        state1; state2; state3; state4;  controls; dcontrols;
    ]

    if DO >= 1
        dstate1_ = astate[DSTATE1_IDX]
        dstate1 = h_prop * (dstate1_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state1_)
        append!(astate_, dstate1)

        dstate2_ = astate[DSTATE2_IDX]
        dstate2 = h_prop * (dstate2_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state2_)
        append!(astate_, dstate2)

        dstate3_ = astate[DSTATE3_IDX]
        dstate3 = h_prop * (dstate3_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state3_)
        append!(astate_, dstate3)

        dstate4_ = astate[DSTATE4_IDX]
        dstate4 = h_prop * (dstate4_ + dt *  NEGI_TRANSMON_NUMBER_ISO * state4_)
        append!(astate_, dstate4)
        # state3_ = astate[STATE3_IDX]
        # state3 = h_prop * state3_
        # state4_ = astate[STATE4_IDX]
        # state4 = h_prop * state4_
        # dstate1_ = astate[DSTATE1_IDX]
        # dstate1 = h_prop * (dstate1_ + dt * NEGI_H0_ISO * state1_)
        # dstate2_ = astate[DSTATE2_IDX]
        # dstate2 = h_prop * (dstate2_ + dt * NEGI_H0_ISO * state2_)
        # dstate3_ = astate[DSTATE3_IDX]
        # dstate3 = h_prop * (dstate3_ + dt * NEGI_H0_ISO * state3_)
        # dstate4_ = astate[DSTATE4_IDX]
        # dstate4 = h_prop * (dstate4_ + dt * NEGI_H0_ISO * state4_)
        # append!(astate_, [state3; state4; dstate1; dstate2; dstate3; dstate4])
    end
    if DO >= 2
        d2state1_ = astate[D2STATE1_IDX]
        d2state1 = h_prop * (d2state1_ + dt * NEGI2_H0_ISO * dstate1_)
        append!(astate_, d2state1)

        d2state2_ = astate[D2STATE2_IDX]
        d2state2 = h_prop * (d2state2_ + dt * NEGI2_H0_ISO * dstate2_)
        append!(astate_, d2state2)

        d2state3_ = astate[D2STATE3_IDX]
        d2state3 = h_prop * (d2state3_ + dt * NEGI2_H0_ISO * dstate3_)
        append!(astate_, d2state3)

        d2state4_ = astate[D2STATE4_IDX]
        d2state4 = h_prop * (d2state4_ + dt * NEGI2_H0_ISO * dstate4_)
        append!(astate_, d2state4)
    end
    return astate_
end
@inline discrete_dynamics!(x_::AbstractVector, ::Type{Q}, model::AbstractModel, x::AbstractVector,
                           u::AbstractVector, t::Real, dt::Real, H::Matrix) where {Q} = (
                               x_ .= RD.discrete_dynamics(Q, model, x, u, t, dt, H)
)

# main
function run_traj(;evolution_time=40., solver_type=altro,
                  sqrtbp=false, derivative_order=0, integrator_type=rk3,
                  qs=[1e0, 1e0, 1e0, 1e-1, 5e-2, 1e-1],
                  smoke_test=false, dt_inv=Int64(5e0), constraint_tol=1e-9, al_tol=1e-5,
                  pn_steps=2, max_penalty=1e12, verbose=true, save=true, max_iterations=Int64(2e5),
                  nf = false, nf_tol = 0., max_cost_value=1e13, benchmark=false, al_iters = 150, pen_in = NaN)
    # model configuration
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.

    # initial state
    x0_ = zeros(n)
    x0_[STATE1_IDX] = IS1_ISO
    x0_[STATE2_IDX] = IS2_ISO
    x0_[STATE3_IDX] = IS3_ISO
    x0_[STATE4_IDX] = IS4_ISO
    # x0_[STATE3_IDX] = IS3_ISO_
    # x0_[STATE4_IDX] = IS4_ISO_
    x0 = SVector{n}(x0_)

    xf_ = zeros(n)
    xf_[STATE1_IDX] =  XPI_G_ISO
    xf_[STATE2_IDX] = XPI_E_ISO
    xf_[STATE3_IDX] = XPI_3_ISO
    xf_[STATE4_IDX] = XPI_4_ISO
    #xf_[CONTROLS_IDX] = [0.,0.]
    # xf_[STATE4_IDX] = gate * IS4_ISO_
    xf = SVector{n}(xf_)

    # control amplitude constraint
    x_max = fill(Inf, n)
    x_max[CONTROLS_IDX] .= A_MAX
    x_max = SVector{n}(x_max)
    x_min = fill(-Inf, n)
    x_min[CONTROLS_IDX] .= -A_MAX
    x_min = SVector{n}(x_min)

    # control amplitude constraint at boundary
    x_max_boundary = fill(Inf, n)
    x_max_boundary[CONTROLS_IDX] .= 0
    x_max_boundary = SVector{n}(x_max_boundary)
    x_min_boundary = fill(-Inf, n)
    x_min_boundary[CONTROLS_IDX] .= 0
    x_min_boundary = SVector{n}(x_min_boundary)

    # initial trajectory
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    ts = zeros(N)
    ts[1] = t0
    for k = 1:N-1
        ts[k + 1] = ts[k] + dt
    end
    U0 = [SVector{m}(
        rand(-A_MAX:A_MAX, CONTROL_COUNT)
    ) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1, ψ2
        #fill(qs[2], 1); # ∫a
        fill(qs[2], CONTROL_COUNT); # a
        fill(qs[3], CONTROL_COUNT); # ∂a
        fill(qs[4], eval(:($derivative_order >= 1 ? $SAMPLE_COUNT * $HDIM_ISO : 0))); # ∂ψ
        fill(qs[5], eval(:($derivative_order >= 2 ? $SAMPLE_COUNT * $HDIM_ISO : 0)));
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[6], CONTROL_COUNT); # ∂2a
    ]))
    objective = LQRObjective(Q, R, Qf, xf, N)

    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)

    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX; STATE2_IDX])
    # must obey unit norm
    nidxs = [STATE1_IDX, STATE2_IDX, STATE3_IDX, STATE4_IDX]
    # if derivative_order >= 1
    #     push!(nidxs, STATE3_IDX)
    #     push!(nidxs, STATE4_IDX)
    # end
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idx) for idx in nidxs]

    #extra_norm = NormConstraint(n,m, 0., TO.Equality(), CONTROLS_IDX)

    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    #add_constraint!(constraints, control_bnd_boundary, N:N)
    add_constraint!(constraints, target_astate_constraint, N:N)
    for norm_constraint in norm_constraints
        add_constraint!(constraints, norm_constraint, 2:N-1)
    end
    #add_constraint!(constraints, extra_norm, N-1:N-1)
    #add_constraint!(constraints, extra_norm, N:N)
    if nf
        nf_sense = nf_tol == 0. ? TO.Equality() : TO.Inequality()
        nf_nopop = NormConstraint(n, m, nf_tol, nf_sense, [STATE1_IDX[3], STATE1_IDX[6]])
        nf_nopop2 = NormConstraint(n, m, nf_tol, nf_sense, [STATE2_IDX[3], STATE2_IDX[6]])
        add_constraint!(constraints, nf_nopop, 2:N-1)
        add_constraint!(constraints, nf_nopop2, 2:N-1)
    end


    # solve problem
    prob = Problem{IT_RDI[integrator_type]}(model, objective, constraints,
                                            x0, xf, Z, N, t0, evolution_time)
    solver = ALTROSolver(prob)
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    projected_newton = true
    constraint_tolerance = solver_type == altro ? constraint_tol : al_tol
    iterations_inner = smoke_test ? 1 : 300
    iterations_outer = smoke_test ? 1 : al_iters
    n_steps = smoke_test ? 1 : pn_steps
    set_options!(solver, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
                 projected_newton_tolerance=al_tol, n_steps=n_steps,
                 penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
                 projected_newton=projected_newton, iterations_inner=iterations_inner,
                 iterations_outer=iterations_outer, iterations=max_iterations,
                 max_cost_value=max_cost_value)
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end

    # post-process
    acontrols_raw = TO.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    Q_raw = Array(Q)
    Q_arr = [Q_raw[i, i] for i in 1:size(Q_raw)[1]]
    Qf_raw = Array(Qf)
    Qf_arr = [Qf_raw[i, i] for i in 1:size(Qf_raw)[1]]
    R_raw = Array(R)
    R_arr = [R_raw[i, i] for i in 1:size(R_raw)[1]]
    cidx_arr = Array(CONTROLS_IDX)
    d2cidx_arr = Array(D2CONTROLS_IDX)
    cmax = TrajectoryOptimization.max_violation(solver)
    cmax_info = TrajectoryOptimization.findmax_violation(TO.get_constraints(solver))
    iterations_ = Altro.iterations(solver)

    result = Dict(
        "acontrols" => acontrols_arr,
        "controls_idx" => cidx_arr,
        "d2controls_dt2_idx" => d2cidx_arr,
        "evolution_time" => evolution_time,
        "astates" => astates_arr,
        "hdim_iso" => HDIM_ISO,
        "Q" => Q_arr,
        "Qf" => Qf_arr,
        "R" => R_arr,
        "ts" => ts,
        "cmax" => cmax,
        "cmax_info" => cmax_info,
        "dt" => dt,
        "derivative_count" => derivative_order,
        "solver_type" => Integer(solver_type),
        "sqrtbp" => Integer(sqrtbp),
        "max_penalty" => max_penalty,
        "constraint_tol" => constraint_tol,
        "al_tol" => al_tol,
        "save_type" => Integer(jl),
        "integrator_type" => Integer(integrator_type),
        "iterations" => iterations_,
        "max_iterations" => max_iterations,
        "pn_steps" => pn_steps,
        "max_cost_value" => max_cost_value,
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
    control_path = generate_file_path("png", "controls", SAVE_PATH)
    plot_controls([save_file_path], control_path)
    data_path = gen_dparam(save_file_path)
    plot_dparam([data_path])
    return result
end


function forward_pass(save_file_path; derivative_order=0, integrator_type=rk3)
    (evolution_time, d2controls, dt
     ) = h5open(save_file_path, "r+") do save_file
         save_type = SaveType(read(save_file, "save_type"))
         if save_type == jl
             d2controls_idx = read(save_file, "d2controls_dt2_idx")
             acontrols = read(save_file, "acontrols")
             d2controls = acontrols[:, d2controls_idx]
             dt = read(save_file, "dt")
             evolution_time = read(save_file, "evolution_time")
         elseif save_type == samplejl
             d2controls = read(save_file, "d2controls_dt2_sample")
             dt = DT_PREF
             ets = read(save_file, "evolution_time_sample")
             evolution_time = Integer(floor(ets / dt)) * dt
         end
         return (evolution_time, d2controls, dt)
     end
    rdi = IT_RDI[integrator_type]
    knot_count = Integer(floor(evolution_time / dt))

    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    time = 0.
    astate = zeros(n)
    astate[STATE1_IDX] = INITIAL_STATE1
    astate[STATE2_IDX] = INITIAL_STATE2
    astate[STATE3_IDX] = INITIAL_STATE3
    astate[STATE4_IDX] = INITIAL_STATE4
    astate = SVector{n}(astate)
    acontrols = [SVector{m}([d2controls[i, 1],]) for i = 1:knot_count - 1]

    for i = 1:knot_count - 1
        astate = SVector{n}(RD.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt))
        time = time + dt
    end

    res = Dict(
        "astate" => astate
    )

    return res
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

function gen_dparam(save_file_path; trial_count=500, sigma_max=1e-4, save=true)
    # grab relevant information
    (evolution_time, dt, derivative_order, U_, integrator_type
     ) = h5open(save_file_path, "r") do save_file
        evolution_time = read(save_file, "evolution_time")
        dt = read(save_file, "dt")
        derivative_order = read(save_file, "derivative_count")
        U_ = read(save_file, "acontrols")
        integrator_type = read(save_file, "integrator_type")
        return (evolution_time, dt, derivative_order, U_,integrator_type)
    end
    # set up problem
    n = size(NEGI_H0_ISO, 1)
    mh1 = zeros(n, n) .= NEGI_H0_ISO
    #Hs = [H for H in (mh1, NEGI_H1R_ISO, NEGI_H1I_ISO)]
    model = Model{derivative_order}()
    n = state_dim(model)
    m = control_dim(model)
    N = Int(floor(evolution_time / dt)) + 1
    U = [U_[k, :] for k = 1:N-1]
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
            discrete_dynamics!(X[k + 1], IT_RDI[IntegratorType(integrator_type)], model, X[k], U[k], ts[k], dt, mh1 )
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
            write(data_file, "save_file_path", save_file_path)
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
