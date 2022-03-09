"""
spin.jl - common definitions for the spin directory
"""
#includes the rbqoc file
WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "rbqoc.jl"))

using Dates
using HDF5
using LinearAlgebra
#test comment
# paths
const TRANSMON_OUT_PATH = abspath(joinpath(WDIR, "out", "transmon"))

# experimental constants
const ω_q = 0 #2π * 4.96 #GHz
const α = -2π * 0.143 #GHz
const TRANSMON_STATE_COUNT = 3

# simulation constants
const DT_PREF = 1e-1

# define the system
const HDIM = TRANSMON_STATE_COUNT
const HDIM_ISO = 2 * HDIM
# operators
const SIGMAX = [0 1;
                1 0]
const SIGMAZ = [1 0;
                0 -1]
# This is placing sqrt(1), ..., sqrt(TRANSMON_STATE_COUNT - 1) on the 1st diagonal
# counted up from the true diagonal.
const TRANSMON_ANNIHILATE = diagm(1 => map(sqrt, 1:TRANSMON_STATE_COUNT-1))
# Taking the adjoint is as simple as adding an apostrophe.
const TRANSMON_CREATE = TRANSMON_ANNIHILATE'
# In julia `a * b` is scalar multiplication where a and b are scalars,
# `a * b` is matrix multiplication where a and b are vectors,
# and `a .* b` is the element-wise product where a and b are vectors.

const TRANSMON_NUMBER = TRANSMON_CREATE * TRANSMON_ANNIHILATE
const NEGI_TRANSMON_NUMBER_ISO = get_mat_iso(-1im * TRANSMON_NUMBER)
const TRANSMON_ID = I(TRANSMON_STATE_COUNT)
const TRANSMON_QUAD = TRANSMON_NUMBER * (TRANSMON_NUMBER - TRANSMON_ID)
const TRANSMON_G = [1; zeros(TRANSMON_STATE_COUNT - 1)]
const TRANSMON_E = [zeros(1); 1; zeros(TRANSMON_STATE_COUNT - 2)]
function transmon_state(level)
    state = zeros(TRANSMON_STATE_COUNT)
    state[level + 1] = 1.
    return state
end

const NEGI_H0_ISO = get_mat_iso(
    -1im * (ω_q * TRANSMON_NUMBER  + α/2 * TRANSMON_QUAD)
)
const NEGI_H1R_ISO = get_mat_iso(
    -1im * (TRANSMON_CREATE + TRANSMON_ANNIHILATE)
)
const NEGI_H1I_ISO = get_mat_iso(
    -1im * ( 1im * (TRANSMON_CREATE - TRANSMON_ANNIHILATE))
)
# gates
Rx(θ) = [cos(θ/2) -1im * sin(θ/2);
         -1im * sin(θ/2) cos(θ/2)]
Rz(θ) = [ℯ^(-1im * θ/2) 0;
         0 ℯ^(1im * θ/2)]
# hamiltonian
# initial states
#Matrix{Float64} is alias for Array{Float64,2}
const ID = Array{Float64,2}(I(HDIM))
const IS1_ISO = get_vec_iso(ID[:,1])
const IS2_ISO = get_vec_iso(ID[:,2])
# target states
const XPIBY2_subspace = hcat([1/sqrt(2); -1im/sqrt(2); zeros(TRANSMON_STATE_COUNT - 2)],
                    [-1im/sqrt(2); 1/sqrt(2); zeros(TRANSMON_STATE_COUNT -2)])
const XPIBY2_ISO = get_vec_iso(XPIBY2_subspace[:,1])
const XPIBY2_ISO2 = get_vec_iso(XPIBY2_subspace[:,2])
const XPI_G_ISO = get_vec_iso([0, -1im, 0])
const XPI_E_ISO = get_vec_iso([-1im, 0, 0])
# paths
const SPIN_OUT_PATH = abspath(joinpath(WDIR, "out", "spin"))
const FBFQ_DFQ_DATA_FILE_PATH = joinpath(SPIN_OUT_PATH, "figures", "misc", "dfq.h5")
@inline fidelity_vec_iso2(s1, s2) = (
    (s1's2)^2 + (s1[1] * s2[3] + s1[2] * s2[4] - s1[3] * s2[1] - s1[4] * s2[2])^2
)


@inline gate_error_iso2a(s1::AbstractVector, s2::AbstractVector;
                         s1o::Int64=0, s2o::Int64=0) = (
    s1[1 + s1o] * s2[1 + s2o] + s1[2 + s1o] * s2[2 + s2o]
    + s1[3 + s1o] * s2[3 + s2o] + s1[4 + s1o] * s2[4 + s2o]
    + s1[5 + s1o] * s2[5 + s2o] + s1[6+s1o] * s2[6 + s2o]
)


@inline gate_error_iso2b(s1::AbstractVector, s2::AbstractVector;
                         s1o::Int64=0, s2o::Int64=0) = (
    -s1[4 + s1o] * s2[1 + s2o] - s1[5 + s1o] * s2[2 + s2o] - s1[6 + s1o] * s2[3 + s2o]
    + s1[1 + s1o] * s2[4 + s2o] + s1[2 + s1o] * s2[5 + s2o] + s1[3 + s1o] * s2[6 + s2o]
)


@inline gate_error_iso2(s1::AbstractVector, s2::AbstractVector;
                        s1o::Int64=0, s2o::Int64=0) = (
    1 - gate_error_iso2a(s1, s2; s1o=s1o, s2o=s2o)^2 - gate_error_iso2b(s1, s2; s1o=s1o, s2o=s2o)^2
)


function jacobian_gate_error_iso2(s1::AbstractVector, s2::AbstractVector;
                                  s1o::Int64=0, s2o::Int64=0)
    a = 2 * gate_error_iso2a(s1, s2; s1o=s1o, s2o=s2o)
    b = 2 * gate_error_iso2b(s1, s2; s1o=s1o, s2o=s2o)
    jac = [
        # -a * s2[1 + s2o] - b * s2[3 + s2o],
        # -a * s2[2 + s2o] - b * s2[4 + s2o],
        # -a * s2[3 + s2o] + b * s2[1 + s2o],
        # -a * s2[4 + s2o] + b * s2[2 + s2o],
        -a * s2[1 + s2o] - b * s2[4 + s2o],
        -a * s2[2 + s2o] - b * s2[5 + s2o],
        -a * s2[3 + s2o] - b * s2[6 + s2o],
        -a * s2[4 + s2o] + b * s2[1 + s2o],
        -a * s2[5 + s2o] + b * s2[2 + s2o],
        -a * s2[6 + s2o] + b * s2[3 + s2o],
    ]
    return jac
end

function hessian_gate_error_iso2(s2::AbstractVector; s2o::Int64=0) where {T}
    # d11 = -2 * s2[1+s2o]^2 - 2 * s2[3+s2o]^2
    # d12 = -2 * s2[1+s2o] * s2[2+s2o] -2 * s2[3+s2o] * s2[4+s2o]
    # d13 = 0
    # d14 = 2 * s2[2+s2o] * s2[3+s2o] - 2 * s2[1+s2o] * s2[4+s2o]
    # d22 = -2 * s2[2+s2o]^2 - 2 * s2[4+s2o]^2
    # d23 = -2 * s2[2+s2o] * s2[3+s2o] + 2 * s2[1+s2o] * s2[4+s2o]
    # d24 = 0
    # d33 = -2 * s2[1+s2o]^2 - 2 * s2[3+s2o]^2
    # d34 = -2 * s2[1+s2o] * s2[2+s2o] - 2 * s2[3+s2o] * s2[4+s2o]
    # d44 = -2 * s2[2+s2o]^2 - 2 * s2[4+s2o]^2
    # hes = [
    #     d11 d12 d13 d14;
    #     d12 d22 d23 d24;
    #     d13 d23 d33 d34;
    #     d14 d24 d34 d44;
    # ]
    d11 = -2 * s2[1+s2o]^2 - 2 * s2[4+s2o]^2
    d12 = -2 * s2[1+s2o] * s2[2+s2o] -2 * s2[4+s2o] * s2[5+s2o]
    d13 = -2 * s2[1 + s2o] * s2[3+s2o] - 2*s2[4+s2o]*s2[6+s2o]
    d14 = 0
    d15 = -2 * s2[1+s2o] * s2[5+s2o] + 2 * s2[4+s2o] * s2[2+s2o]
    d16 = -2 * s2[1+s2o] * s2[6 + s2o] + 2 * s2[4 + s2o] * s2[3+s2o]
    d22 = -2 * s2[2+s2o]^2 - 2 * s2[5+s2o]^2
    d23 = -2 * s2[2+s2o] * s2[3+s2o] - 2 * s2[5+s2o] * s2[6+s2o]
    d24 =  -2 * s2[2+s2o] * s2[4+s2o] + 2 * s2[5+s2o] * s2[1+s2o]
    d25 = 0
    d26 = 2 * s2[2+s2o] * s2[4+s2o] - 2 * s2[5+s2o] * s2[1+s2o]
    d33 = -2 * s2[6+s2o]^2 - 2 * s2[3+s2o]^2
    d34 = -2 * s2[3+s2o] * s2[4+s2o] + 2 * s2[1+s2o] * s2[4+s2o]
    d35 = -2 * s2[3+s2o] *  s2[5+s2o] + 2 * s2[6+s2o] * s2[2+s2o]
    d36 = 0
    d44 = -2 * s2[1+s2o]^2 - 2 * s2[4+s2o]^2
    d45 = -2 * s2[1+s2o] *  s2[2+s2o] - 2 * s2[4+s2o] * s2[5+s2o]
    d46 =  -2 * s2[1+s2o] *  s2[3+s2o] - 2 * s2[4+s2o] * s2[6+s2o]
    d55 = -2 * s2[5+s2o]^2 - 2 * s2[2+s2o]^2
    d56 = -2 * s2[5 + s2o] * s2[6 + s2o] - 2 * s2[2 + s2o] * s2[3+s2o]
    d66 = -2 * s2[6+s2o]^2 - 2 * s2[3+s2o]^2
    hes = [
        d11 d12 d13 d14 d15 d16;
        d12 d22 d23 d24 d25 d26;
        d13 d23 d33 d34 d35 d36;
        d14 d24 d34 d44 d45 d46;
        d15 d25 d35 d45 d55 d56;
        d16 d26 d36 d46 d56 d66;
    ]
    return hes
end

"""
Schroedinger dynamics.
function dynamics_schroed_deqjl(state::SVector, params::SimParams, time::Float64)
    controls_knot_point = (Int(floor(time * params.controls_dt_inv)) % params.control_knot_count) + 1
    negi_h = (
        params.negi_h0
        + params.controls[controls_knot_point][1] * NEGI_H1_ISO
    )
    return (
        negi_h * state
    )
end


function integrate_prop_schroed!(gate_count::Int, states::Array{T, 2},
                                 state::SVector, params::SimParams) where {T}
    states[1, :] = state
    dt = params.controls_dt_inv^(-1)
    prop = I(HDIM_ISO)
    for j = 1:params.control_knot_count
        hamiltonian = params.negi_h0 + params.controls[j, 1] * NEGI_H1_ISO
        prop_ = exp(hamiltonian * dt)
        prop = prop_ * prop
    end
    for i = 2:gate_count + 1
        state = prop * state
        states[i, :] = state
    end
end


function dynamics_schroedda_deqjl(state::SVector, params::SimParams, time::Float64)
    control_knot_point = (Int(floor(time * params.controls_dt_inv)) % params.control_knot_count) + 1
    noise_knot_point = Int(floor(time * params.noise_dt_inv)) + 1
    delta_a = params.noise_offsets[noise_knot_point]
    negi_h = (
        FQ_NEGI_H0_ISO
        + (params.controls[control_knot_point][1] + delta_a) * NEGI_H1_ISO
    )
    return (
        negi_h * state
    )
end


function integrate_prop_schroedda!(gate_count::Int, states::Array{T, 2},
                                   state::SVector, params::SimParams) where {T}
    dt = params.controls_dt_inv^(-1)
    time = 0.
    states[1, :] = state
    for i = 2:gate_count + 1
        for j = 1:params.control_knot_count
            delta_a = params.noise_offsets[Int(floor(time * params.noise_dt_inv)) + 1]
            hamiltonian = params.negi_h0 + (params.controls[j, 1] + delta_a) * NEGI_H1_ISO
            unitary = exp(hamiltonian * dt)
            state = unitary * state
            time = time + dt
        end
        states[i, :] = state
    end
end

compute_fidelities

function compute_fidelities(gate_count, gate_type, states::Array{T, N}) where {T, N}
    # Compute the fidelities.
    # All of the gates we consider are 4-cyclic up to phase.
    state_type = length(size(states)) == 2 ? st_state : st_density
    initial_state = state_type == st_state ? states[1, :] : states[1, :, :]
    fidelities = zeros(gate_count + 1)
    is_iso = !(states[1] isa Complex)
    g1 = is_iso ? GT_GATE_ISO[gate_type] : GT_GATE[gate_type]
    g2 = g1^2
    g3 = g1^3
    id0 = initial_state
    if state_type == st_state
        id1 = g1 * id0
        id2 = g2 * id0
        id3 = g3 * id0
    elseif state_type == st_density
        id1 = g1 * id0 * g1'
        id2 = g2 * id0 * g2'
        id3 = g3 * id0 * g3'
    end
    # Compute the fidelity after each gate.
    for i = 1:gate_count + 1
        # 1-indexing means we are 1 ahead for modulo arithmetic.
        i_eff = i - 1
        if i_eff % 4 == 0
            target = id0
        elseif i_eff % 4 == 1
            target = id1
        elseif i_eff % 4 == 2
            target = id2
        elseif i_eff % 4 == 3
            target = id3
        end
        if state_type == st_state
            fidelities[i] = fidelity_vec_iso2(states[i, :], target)
        elseif state_type == st_density
            fidelities[i] = (is_iso ? fidelity_mat_iso(states[i, :, :], target)
                             : fidelity_mat(states[i, :, :], target))
        end
    end

    return fidelities
end

function evaluate_fqdev(;fq_cov=FQ * 1e-2, trial_count=1000,
                        save_file_path=nothing, dynamics_type=schroed,
                        gate_type=zpiby2)
    negi_hp = (FQ + fq_cov) * NEGI_H0_ISO
    negi_hn = (FQ - fq_cov) * NEGI_H0_ISO
    dynamics_type = schroed
    gate_errors = zeros(2 * trial_count)
    gate_errors_basis = zeros(2 * 4)

    for i = -4:-1
        res1 = run_sim_prop(
            1, gate_type; save_file_path=save_file_path, dynamics_type=dynamics_type,
            negi_h0=negi_hp, save=false, seed=0, state_seed=i,
        )
        res2 = run_sim_prop(
            1, gate_type; save_file_path=save_file_path, dynamics_type=dynamics_type,
            negi_h0=negi_hn, save=false, seed=0, state_seed=i,
        )
        ge1 = 1 - res1["fidelities"][end]
        ge2 = 1 - res2["fidelities"][end]
        gate_errors_basis[2 * (i + 4) + 1] = ge1
        gate_errors_basis[2 * (i + 4) + 2] = ge2
    end

    for i = 0:trial_count - 1
        res1 = run_sim_prop(
            1, gate_type; save_file_path=save_file_path, dynamics_type=dynamics_type,
            negi_h0=negi_hp, save=false, seed=i
        )
        res2 = run_sim_prop(
            1, gate_type; save_file_path=save_file_path, dynamics_type=dynamics_type,
            negi_h0=negi_hn, save=false, seed=i
        )
        ge1 = 1 - res1["fidelities"][end]
        ge2 = 1 - res2["fidelities"][end]
        gate_errors[2 * i + 1] = ge1
        gate_errors[2 * i + 2] = ge2
    end

    result = Dict(
        "gate_errors" => gate_errors,
        "gate_errors_basis" => gate_errors_basis,
    )

    return result
end


function evaluate_adev(;gate_count=100, gate_type=xpiby2, dynamics_type=schroedda,
                       save_file_path=nothing, trial_count=100)
    gate_errors = zeros(trial_count)
    for seed = 1:trial_count
        res = run_sim_prop(gate_count, gate_type; dynamics_type=dynamics_type,
                           save_file_path=save_file_path, seed=seed, save=false)
        gate_error = 1 - res["fidelities"][end]
        gate_errors[seed] = gate_error
    end

    result = Dict(
        "gate_errors" => gate_errors,
    )
end
"""
function read_final_state(data_path)
    state = h5open(data_path , "r") do data_file
        state = read(data_file, "astates")
        return state
    end
    return get_vec_uniso(state[end, 1:HDIM_ISO])
end
