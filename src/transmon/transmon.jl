"""
transmon.jl - sampling robustness for the δω problem

This optimization uses the infidelity metric rather than
the standard diagonal LQR metric.
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "transmon", "system.jl"))

using Altro
using HDF5
using LinearAlgebra
using RobotDynamics
using StaticArrays
using TrajectoryOptimization
const RD = RobotDynamics
const TO = TrajectoryOptimization

# paths
const EXPERIMENT_META = "transmon"
const EXPERIMENT_NAME = "transmon_sample"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

# problem
const A_MAX = 2π * 6e-3
const CONTROL_COUNT = 2
const STATE_COUNT = 1
const ASTATE_SIZE_BASE = STATE_COUNT * HDIM_ISO + 2 * CONTROL_COUNT
const SAMPLE_COUNT = 2
const ASTATE_SIZE = ASTATE_SIZE_BASE + SAMPLE_COUNT * HDIM_ISO
const ACONTROL_SIZE = CONTROL_COUNT
# state indices
const STATE1_IDX = 1:HDIM_ISO
const CONTROLS_IDX = STATE1_IDX[end] + 1:STATE1_IDX[end] + CONTROL_COUNT
const DCONTROLS_IDX = CONTROLS_IDX[end] + 1:CONTROLS_IDX[end] + CONTROL_COUNT
const S1_IDX = DCONTROLS_IDX[end] + 1:DCONTROLS_IDX[end] + HDIM_ISO
const S2_IDX = S1_IDX[end] + 1:S1_IDX[end] + HDIM_ISO
# const S3_IDX = S2_IDX[end] + 1:S2_IDX[end] + HDIM_ISO
# const S4_IDX = S3_IDX[end] + 1:S3_IDX[end] + HDIM_ISO
# const S5_IDX = S4_IDX[end] + 1:S4_IDX[end] + HDIM_ISO
# const S6_IDX = S5_IDX[end] + 1:S5_IDX[end] + HDIM_ISO
# const S7_IDX = S6_IDX[end] + 1:S6_IDX[end] + HDIM_ISO
# const S8_IDX = S7_IDX[end] + 1:S7_IDX[end] + HDIM_ISO
# control indices
const D2CONTROLS_IDX = 1:CONTROL_COUNT

# model
struct Model <: AbstractModel
    h0_samples::Vector{SMatrix{HDIM_ISO, HDIM_ISO}}
end
@inline RD.state_dim(::Model) = ASTATE_SIZE
@inline RD.control_dim(::Model) = ACONTROL_SIZE


# This cost puts a gate error cost on
# the sample states and a LQR cost on the other terms.
# The hessian w.r.t the state and controls is constant.
struct Cost{N,M,T} <: TO.CostFunction
    Q::Diagonal{T, SVector{N,T}}
    R::Diagonal{T, SVector{M,T}}
    q::SVector{N, T}
    c::T
    hess_astate::Symmetric{T, SMatrix{N,N,T}}
    target_states::Array{SVector{HDIM_ISO, T}, 1}
    q_ss1::T
    q_ss2::T
    q_ss3::T
    q_ss4::T
end

function Cost(Q::Diagonal{T,SVector{N,T}}, R::Diagonal{T,SVector{M,T}},
              xf::SVector{N,T}, target_states::Array{SVector{HDIM_ISO}, 1},
              q_ss1::T, q_ss2::T, q_ss3::T, q_ss4::T) where {N,M,T}
    q = -Q * xf
    c = 0.5 * xf' * Q * xf
    hess_astate = zeros(N, N)
    # For reasons unknown to the author, throwing a -1 in front
    # of the gate error Hessian makes the cost function work.
    # This is strange, because the gate error Hessian has been
    # checked against autodiff.
    hess_state1 = -1 * q_ss1 * hessian_gate_error_iso2(target_states[1])

    hess_astate[S1_IDX, S1_IDX] = hess_state1
    hess_astate[S2_IDX, S2_IDX] = hess_state1
    hess_astate += Q
    hess_astate = Symmetric(SMatrix{N, N}(hess_astate))
    return Cost{N,M,T}(Q, R, q, c, hess_astate, target_states, q_ss1, q_ss2, q_ss3, q_ss4)
end

@inline TO.state_dim(cost::Cost{N,M,T}) where {N,M,T} = N
@inline TO.control_dim(cost::Cost{N,M,T}) where {N,M,T} = M
@inline Base.copy(cost::Cost{N,M,T}) where {N,M,T} = Cost{N,M,T}(
    cost.Q, cost.R, cost.q, cost.c, cost.hess_astate,
    cost.target_states, cost.q_ss1, cost.q_ss2, cost.q_ss3, cost.q_ss4
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N}) where {N,M,T} = (
    0.5 * astate' * cost.Q * astate + cost.q'astate + cost.c
    + cost.q_ss1 * gate_error_iso2(astate, cost.target_states[1]; s1o=S1_IDX[1] - 1)
    + cost.q_ss1 * gate_error_iso2(astate, cost.target_states[1]; s1o=S2_IDX[1] - 1)
)

@inline TO.stage_cost(cost::Cost{N,M,T}, astate::SVector{N},
                      acontrol::SVector{M}) where {N,M,T} = (
    TO.stage_cost(cost, astate) + 0.5 * acontrol' * cost.R * acontrol
)

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T},
                      astate::SVector{N,T}) where {N,M,T}
    E.q = (cost.Q * astate + cost.q + [
        @SVector zeros(ASTATE_SIZE_BASE);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_states[1]; s1o=S1_IDX[1] - 1);
        cost.q_ss1 * jacobian_gate_error_iso2(astate, cost.target_states[1]; s1o=S2_IDX[1] - 1);
    ])
    return false
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                      acontrol::SVector{M,T}) where {N,M,T}
    TO.gradient!(E, cost, astate)
    E.r = cost.R * acontrol
    E.c = 0
    return false
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T}) where {N,M,T}
    E.Q = cost.hess_astate
    return true
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::Cost{N,M,T}, astate::SVector{N,T},
                     acontrol::SVector{M,T}) where {N,M,T}
    TO.hessian!(E, cost, astate)
    E.R = cost.R
    E.H .= 0
    return true
end


# dynamics
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::StaticVector,
                              acontrols::StaticVector, time::Real, dt::Real) where {SC}
    negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1R_ISO + astate[CONTROLS_IDX[2]] * NEGI_H1I_ISO
    h_prop = exp((NEGI_H0_ISO + negi_hc) * dt)
    state1 = h_prop * astate[STATE1_IDX]
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] .* dt
    dcontrols = astate[DCONTROLS_IDX] + acontrols[D2CONTROLS_IDX] .* dt

    hp_prop = exp((model.h0_samples[1] + negi_hc) * dt)
    hn_prop = exp((model.h0_samples[2] + negi_hc) * dt)
    s1 = hp_prop * astate[S1_IDX]
    s2 = hn_prop * astate[S2_IDX]

    astate_ = [
        state1; controls; dcontrols;
        s1; s2;
    ]

    return astate_
end
function RD.discrete_dynamics(::Type{RK3}, model::Model, astate::AbstractVector,
                              acontrols::AbstractVector, time::Real, dt::Real, H::Matrix)
    negi_hc = astate[CONTROLS_IDX[1]] * NEGI_H1R_ISO + astate[CONTROLS_IDX[2]] * NEGI_H1I_ISO
    h_prop = exp((H + negi_hc) * dt)
    state1 = h_prop * astate[STATE1_IDX]
    controls = astate[CONTROLS_IDX] + astate[DCONTROLS_IDX] .* dt
    dcontrols = astate[DCONTROLS_IDX] + acontrols[D2CONTROLS_IDX] .* dt

    hp_prop = exp((model.h0_samples[1] + negi_hc) * dt)
    hn_prop = exp((model.h0_samples[2] + negi_hc) * dt)
    s1 = hp_prop * astate[S1_IDX]
    s2 = hn_prop * astate[S2_IDX]

    astate_ = [
        state1; controls; dcontrols;
        s1; s2;
    ]

    return astate_
end
@inline discrete_dynamics!(x_::AbstractVector, ::Type{Q}, model::AbstractModel, x::AbstractVector,
                           u::AbstractVector, t::Real, dt::Real, H::Matrix) where {Q} = (
                               x_ .= RD.discrete_dynamics(Q, model, x, u, t, dt, H)
)
# main
function run_traj(;evolution_time=12., solver_type=altro,
                  sqrtbp=false, integrator_type=rk3,
                  qs=[1e0, 1e0, 1e0, 1e-1, 1e0, 1e0, 1e0, 1e0, 1e-1],
                  dt_inv=Int64(1e1), smoke_test=false, constraint_tol=1e-8, al_tol=1e-4,
                  pn_steps=2, max_penalty=1e11, verbose=true, save=true,
                  max_iterations=Int64(2e5), ω_cov= 2π * 2e-2, benchmark=false,
                  nf = false, nf_tol = 0., max_cost_value=1e8)
    # model configuration
    h0_samples = Array{SMatrix{HDIM_ISO, HDIM_ISO}}(undef, SAMPLE_COUNT)
    h0_samples[1] = get_mat_iso(
        -1im * (ω_cov * TRANSMON_NUMBER  + α/2 * TRANSMON_QUAD)
    )
    h0_samples[2] = get_mat_iso(
        -1im * (-ω_cov * TRANSMON_NUMBER  + α/2 * TRANSMON_QUAD)
    )
    model = Model(h0_samples)
    n = state_dim(model)
    m = control_dim(model)
    t0 = 0.
    tf = evolution_time
    dt = dt_inv^(-1)
    N = Int(floor(evolution_time * dt_inv)) + 1
    ts = zeros(N)
    ts[1] = t0
    for k = 1:N-1
        ts[k + 1] = ts[k] + dt
    end
    # initial state
    x0 = SVector{n}([
        IS1_ISO;
        zeros(2 * CONTROL_COUNT);
        repeat(IS1_ISO, 2);
    ])

    # final state
    target_states = Array{SVector{HDIM_ISO}, 1}(undef, 1)
    target_states[1] = XPIBY2_ISO
    xf = SVector{n}([
        XPIBY2_ISO;
        zeros(2 * CONTROL_COUNT);
        repeat(XPIBY2_ISO, 2);
    ])

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
    U0 = [SVector{m}([
        fill(1e-6, CONTROL_COUNT);
    ]) for k = 1:N-1]
    X0 = [SVector{n}([
        fill(NaN, n);
    ]) for k = 1:N]
    Z = Traj(X0, U0, dt * ones(N))

    # cost function
    Q = Diagonal(SVector{n}([
        fill(qs[1], STATE_COUNT * HDIM_ISO); # ψ1, ψ2
        fill(qs[2], CONTROL_COUNT); # a
        fill(qs[3], CONTROL_COUNT); # ∂a
        fill(1e3, SAMPLE_COUNT * HDIM_ISO);
    ]))
    Qf = Q * N
    R = Diagonal(SVector{m}([
        fill(qs[9], CONTROL_COUNT); # ∂2a
    ]))
    # objective = LQRObjective(Q, R, Qf, xf, N)
    cost_k = Cost(Q, R, xf, target_states, qs[5], qs[6], qs[7], qs[8])
    cost_f = Cost(Qf, R, xf, target_states, N * qs[5], N * qs[6], N * qs[7], N * qs[8])
    objective = TO.Objective(cost_k, cost_f, N)

    # must satisfy control amplitude bound
    control_bnd = BoundConstraint(n, m, x_max=x_max, x_min=x_min)
    # must statisfy conrols start and end at 0
    control_bnd_boundary = BoundConstraint(n, m, x_max=x_max_boundary, x_min=x_min_boundary)
    # must reach target state, must have zero net flux
    target_astate_constraint = GoalConstraint(xf, [STATE1_IDX;])
    # must obey unit norm.
    norm_constraints = [NormConstraint(n, m, 1, TO.Equality(), idxs) for idxs in (
        STATE1_IDX, S1_IDX, S2_IDX
    )]
    constraints = ConstraintList(n, m, N)
    add_constraint!(constraints, control_bnd, 2:N-2)
    add_constraint!(constraints, control_bnd_boundary, N-1:N-1)
    add_constraint!(constraints, target_astate_constraint, N:N);
    for norm_constraint in norm_constraints
        add_constraint!(constraints, norm_constraint, 2:N-1)
    end
    if nf
        nf_sense = nf_tol == 0. ? TO.Equality() : TO.Inequality()
        nf_nopop = NormConstraint(n, m, nf_tol, nf_sense, [3, 3 + HDIM_ISO])
        add_constraint!(constraints, nf_nopop, 2:N-1)
    end
    # solve problem
    prob = Problem{IT_RDI[integrator_type]}(model, objective, constraints,
                                            x0, xf, Z, N, t0, evolution_time)
    solver = ALTROSolver(prob)
    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    projected_newton = solver_type == altro ? true : false
    constraint_tolerance = solver_type == altro ? constraint_tol : al_tol
    iterations_inner = smoke_test ? 1 : 300
    iterations_outer = smoke_test ? 1 : 30
    n_steps = smoke_test ? 1 : pn_steps
    set_options!(solver, square_root=sqrtbp, constraint_tolerance=constraint_tolerance,
                 projected_newton_tolerance=al_tol, n_steps=n_steps,
                 penalty_max=max_penalty, verbose_pn=verbose_pn, verbose=verbose_,
                 projected_newton=projected_newton, iterations_inner=iterations_inner,
                 iterations_outer=iterations_outer, iterations=max_iterations)
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
    cmax = TO.max_violation(solver)
    cmax_info = TO.findmax_violation(TO.get_constraints(solver))
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
        "transmon_state_count" => TRANSMON_STATE_COUNT,
        "ω_cov" => ω_cov
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


function forward_pass(save_file_path; integrator_type=rk6, gate_type=xpiby2)
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

    if gate_type == xpiby2
        target_state1 = Array(XPIBY2_ISO_1)
        target_state2 = Array(XPIBY2_ISO_2)
    elseif gate_type == ypiby2
        target_state1 = Array(YPIBY2_ISO_1)
        target_state2 = Array(YPIBY2_ISO_2)
    elseif gate_type == zpiby2
        target_state1 = Array(ZPIBY2_ISO_1)
        target_state2 = Array(ZPIBY2_ISO_2)
    end

    model = Model(sample_count)
    n = state_dim(model)
    m = control_dim(model)
    time = 0.
    astate = SVector{n}([
        IS1;
        IS2;
        zeros(3 * CONTROL_COUNT);
        repeat([IS1; IS2], sample_count);
    ])
    acontrols = [SVector{m}([d2controls[i, 1],]) for i = 1:knot_count - 1]

    for i = 1:knot_count - 1
        astate = RD.discrete_dynamics(rdi, model, astate, acontrols[i], time, dt)
        time = time + dt
    end

    res = Dict(
        "astate" => astate,
        "target_state1" => target_state1,
        "target_state2" => target_state2,
    )

    return res
end


function state_diffs(save_file_path; gate_type=zpiby2)
    (astates,
     ) = h5open(save_file_path, "r") do save_file
        astates = read(save_file, "astates")
        return (astates,)
    end
    knot_count = size(astates, 1)
    fidelities = zeros(SAMPLE_COUNT)
    mse = zeros(SAMPLE_COUNT)
    gate_unitary = GT_GATE[gate_type]
    ts1 = gate_unitary * IS1
    ts2 = gate_unitary * IS2
    ts3 = gate_unitary * IS3
    ts4 = gate_unitary * IS4
    s1 = astates[end, S1_IDX]
    fidelities[1] = fidelity_vec_iso2(s1, ts1)
    d1 = s1 - ts1
    mse[1] = d1'd1
    s2 = astates[end, S2_IDX]
    fidelities[2] = fidelity_vec_iso2(s2, ts2)
    d2 = s2 - ts2
    mse[2] = d2'd2
    s3 = astates[end, S3_IDX]
    fidelities[3] = fidelity_vec_iso2(s3, ts3)
    d3 = s3 - ts3
    mse[3] = d3'd3
    s4 = astates[end, S4_IDX]
    fidelities[4] = fidelity_vec_iso2(s4, ts4)
    d4 = s4 - ts4
    mse[4] = d4'd4
    s5 = astates[end, S5_IDX]
    fidelities[5] = fidelity_vec_iso2(s5, ts1)
    d5 = s5 - ts1
    mse[5] = d5'd5
    s6 = astates[end, S6_IDX]
    fidelities[6] = fidelity_vec_iso2(s6, ts2)
    d6 = s6 - ts2
    mse[6] = d6'd6
    s7 = astates[end, S7_IDX]
    fidelities[7] = fidelity_vec_iso2(s7, ts3)
    d7 = s7 - ts3
    mse[7] = d7'd7
    s8 = astates[end, S8_IDX]
    fidelities[8] = fidelity_vec_iso2(s8, ts4)
    d8 = s8 - ts4
    mse[8] = d8'd8

    return (fidelities, mse)
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
    (evolution_time, dt, U_, integrator_type, ω
     ) = h5open(save_file_path, "r") do save_file
        evolution_time = read(save_file, "evolution_time")
        dt = read(save_file, "dt")
        U_ = read(save_file, "acontrols")
        integrator_type = read(save_file, "integrator_type")
        ω = read(save_file, "ω_cov")
        return (evolution_time, dt, U_,integrator_type, ω)
    end
    # set up problem
    n = size(NEGI_H0_ISO, 1)
    mh1 = zeros(n, n) .= NEGI_H0_ISO
    #Hs = [H for H in (mh1, NEGI_H1R_ISO, NEGI_H1I_ISO)]
    h0_samples = Array{SMatrix{HDIM_ISO, HDIM_ISO}}(undef, SAMPLE_COUNT)
    h0_samples[1] = get_mat_iso(
        -1im * (ω * TRANSMON_NUMBER  + α/2 * TRANSMON_QUAD)
    )
    h0_samples[2] = get_mat_iso(
        -1im * (-ω * TRANSMON_NUMBER  + α/2 * TRANSMON_QUAD)
    )
    model = Model(h0_samples)
    n = state_dim(model)
    m = control_dim(model)
    N = Int(floor(evolution_time / dt)) + 1
    U = [U_[k, :] for k = 1:N-1]
    X = [zeros(n) for i = 1:N]
    ts = [dt * (k - 1) for k = 1:N]
    # initial state
    x0 = zeros(n)
    x0[STATE1_IDX] = IS1_ISO
    X[1] = x0
    # target state
    # xf = zeros(n)
    # cavity_state_ = cavity_state(target_level)
    # ψT = kron(cavity_state_, TRANSMON_G)
    # xf[model.state1_idx] = get_vec_iso(ψT)
    # xf = V(xf)
    xf = zeros(n)
    ψT = XPIBY2_subspace[:,1]
    xf[STATE1_IDX] = XPIBY2_ISO
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
            discrete_dynamics!(X[k + 1], IT_RDI[IntegratorType(integrator_type)], model, X[k], U[k], ts[k], dt, mh1 )
        end
        ψN = get_vec_uniso(X[N][STATE1_IDX])
        gate_error = 1 - abs(ψT'ψN)^2
        gate_errors[i] = gate_error
    end
    # save
    data_file_path = generate_file_path("h5", "dparams", SAVE_PATH)
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
    #log_ge = map(x -> log10(x), gate_errors__)
    label = isnothing(labels) ? nothing : labels[i]
    Plots.plot!(fracs__.*(1e3/2π), gate_errors__, label=label)
    plot_file_path = generate_file_path("png", "gate_error_plot", SAVE_PATH)
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end
