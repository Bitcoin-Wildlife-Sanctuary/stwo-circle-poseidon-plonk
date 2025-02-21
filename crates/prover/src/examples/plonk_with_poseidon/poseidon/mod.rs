#![allow(unused)]

use std::cmp::max;
use std::ops::{Add, AddAssign, Mul, Neg, Sub};

use itertools::Itertools;
use num_traits::{One, Zero};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::{span, Level};

use crate::constraint_framework::logup::LogupTraceGenerator;
use crate::constraint_framework::preprocessed_columns::PreProcessedColumnId;
use crate::constraint_framework::{
    assert_constraints, EvalAtRow, FrameworkComponent, FrameworkEval, Relation, RelationEntry,
    TraceLocationAllocator,
};
use crate::core::backend::simd::m31::{PackedBaseField, PackedM31, LOG_N_LANES, N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{BackendForChannel, Col, Column};
use crate::core::channel::MerkleChannel;
use crate::core::fields::m31::{pow2147483645, BaseField, M31};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::fields::FieldExpOps;
use crate::core::pcs::{CommitmentSchemeProver, PcsConfig};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, StarkProof};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::vcs::poseidon31_ref::{
    FIRST_FOUR_ROUND_RC, LAST_FOUR_ROUNDS_RC, MAT_DIAG16_M_1, PARTIAL_ROUNDS_RC,
};
use crate::core::ColumnVec;
use crate::examples::plonk_with_poseidon::plonk::PlonkWithAcceleratorLookupElements;

/// Preprocessed columns for describing a plonk circuit.
/// Each plonk gate is described by input wires `a_wire`, `b_wire`, output wire `c_wire`, and
/// operation `op`.  
#[derive(Debug)]
pub struct Poseidon {
    pub name: String,
}
impl Poseidon {
    pub const fn new(name: String) -> Self {
        Self { name }
    }

    pub fn id(&self) -> PreProcessedColumnId {
        PreProcessedColumnId {
            id: format!("preprocessed_poseidon_{}", self.name),
        }
    }
}

const N_STATE: usize = 16;
const N_HALF_FULL_ROUNDS: usize = 4;
const N_PARTIAL_ROUNDS: usize = 14;
const FULL_ROUNDS: usize = 2 * N_HALF_FULL_ROUNDS;
const N_COLUMNS: usize = N_STATE * 3;
const LOG_EXPAND: u32 = 3;

pub const CONSTANT_1: [M31; 8] = [M31::from_u32_unchecked(0); 8];
pub const CONSTANT_2: [M31; 8] = [
    M31::from_u32_unchecked(0x7412ba68),
    M31::from_u32_unchecked(0x70d9f6f6),
    M31::from_u32_unchecked(0x045f05e1),
    M31::from_u32_unchecked(0x056a10e6),
    M31::from_u32_unchecked(0x4aeaffae),
    M31::from_u32_unchecked(0x6b01895d),
    M31::from_u32_unchecked(0x20f34b05),
    M31::from_u32_unchecked(0x7d6e4f58),
];
pub const CONSTANT_3: [M31; 8] = [
    M31::from_u32_unchecked(0x16d7f425),
    M31::from_u32_unchecked(0x5bed6a76),
    M31::from_u32_unchecked(0x50fb5a7d),
    M31::from_u32_unchecked(0x4f86ae48),
    M31::from_u32_unchecked(0x72d1a80d),
    M31::from_u32_unchecked(0x710419a6),
    M31::from_u32_unchecked(0x28679230),
    M31::from_u32_unchecked(0x249e2073),
];

pub type PoseidonAcceleratorComponent = FrameworkComponent<PoseidonAcceleratorEval>;

#[derive(Clone)]
pub struct PoseidonAcceleratorEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkWithAcceleratorLookupElements,
    pub total_sum: SecureField,
}

impl FrameworkEval for PoseidonAcceleratorEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }
    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + LOG_EXPAND
    }
    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        eval_poseidon_constraints(&mut eval, &self.lookup_elements);
        eval
    }
}

#[inline(always)]
/// Applies the M4 MDS matrix described in <https://eprint.iacr.org/2023/323.pdf> 5.1.
fn apply_m4<F>(x: [F; 4]) -> [F; 4]
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let t0 = x[0].clone() + x[1].clone();
    let t02 = t0.clone() + t0.clone();
    let t1 = x[2].clone() + x[3].clone();
    let t12 = t1.clone() + t1.clone();
    let t2 = x[1].clone() + x[1].clone() + t1.clone();
    let t3 = x[3].clone() + x[3].clone() + t0.clone();
    let t4 = t12.clone() + t12.clone() + t3.clone();
    let t5 = t02.clone() + t02.clone() + t2.clone();
    let t6 = t3.clone() + t5.clone();
    let t7 = t2.clone() + t4.clone();
    [t6, t5, t7, t4]
}

/// Applies the external round matrix.
/// See <https://eprint.iacr.org/2023/323.pdf> 5.1 and Appendix B.
fn apply_external_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    // Applies circ(2M4, M4, M4, M4).
    for i in 0..4 {
        [
            state[4 * i],
            state[4 * i + 1],
            state[4 * i + 2],
            state[4 * i + 3],
        ] = apply_m4([
            state[4 * i].clone(),
            state[4 * i + 1].clone(),
            state[4 * i + 2].clone(),
            state[4 * i + 3].clone(),
        ]);
    }
    for j in 0..4 {
        let s =
            state[j].clone() + state[j + 4].clone() + state[j + 8].clone() + state[j + 12].clone();
        for i in 0..4 {
            state[4 * i + j] += s.clone();
        }
    }
}

// Applies the internal round matrix.
//   mu_i = 2^{i+1} + 1.
// See <https://eprint.iacr.org/2023/323.pdf> 5.2.
fn apply_internal_round_matrix<F>(state: &mut [F; 16])
where
    F: Clone + AddAssign<F> + Add<F, Output = F> + Sub<F, Output = F> + Mul<BaseField, Output = F>,
{
    let sum = state[1..]
        .iter()
        .cloned()
        .fold(state[0].clone(), |acc, s| acc + s);

    state[0] += state[0].clone() + state[0].clone() + sum.clone();
    state.iter_mut().enumerate().skip(1).for_each(|(i, s)| {
        // TODO(andrew): Change to rotations.
        *s = s.clone() * BaseField::from_u32_unchecked(1 << (i + 1)) + sum.clone();
    });
}

fn pow5<F: FieldExpOps>(x: F) -> F {
    let x2 = x.clone() * x.clone();
    let x4 = x2.clone() * x2.clone();
    x4 * x.clone()
}

pub fn eval_poseidon_constraints<E: EvalAtRow>(
    eval: &mut E,
    lookup_elements: &PlonkWithAcceleratorLookupElements,
) {
    // preprocessed columns:
    //     is_first_round
    //     is_last_round
    //     is_full_round
    //     round_id
    //     rc0 (16 elements)
    //       -> first element used for swap_bit_addr
    //     rc1 (16 elements)
    //     external_idx_1 (default = 0)
    //     external_idx_2 (default = 0)
    //     is_external_idx_1_nonzero
    //     is_external_idx_2_nonzero
    //
    // trace columns:
    //     in_state (16 elements)
    //     intermediate_state (16 elements)
    //       -> first element used for swap_bit_val
    //     out_state (16 elements)
    //
    // dummy column:
    //     is_first_round = 1
    //     is_last_round = 1
    //     is_full_round = 0
    //     round_id = 0
    //     rc0 (16 elements) = 0
    //     rc1 (16 elements) = 0
    //     external_idx_1 = 0
    //     external_idx_2 = 0
    //     is_external_idx_1_nonzero = 0
    //     is_external_idx_1_nonzero = 0
    //
    //     in_state (16 elements) = 0
    //     intermediate_state (16 elements) = 0
    //     out_state (16 elements) = 0

    let is_first_round =
        eval.get_preprocessed_column(Poseidon::new("is_first_round".to_string()).id());
    let is_last_round =
        eval.get_preprocessed_column(Poseidon::new("is_last_round".to_string()).id());
    let is_full_round =
        eval.get_preprocessed_column(Poseidon::new("is_full_round".to_string()).id());

    let is_not_first_round = E::F::one() - is_first_round.clone();
    let is_not_last_round = E::F::one() - is_last_round.clone();
    let is_partial_round = is_not_first_round.clone() - is_full_round.clone();

    let round_id = eval.get_preprocessed_column(Poseidon::new("round_id".to_string()).id());

    let mut rc0 = vec![];
    for i in 0..16 {
        rc0.push(
            eval.get_preprocessed_column(Poseidon::new(format!("rc0 {}", i).to_string()).id()),
        );
    }
    let mut rc1 = vec![];
    for i in 0..16 {
        rc1.push(
            eval.get_preprocessed_column(Poseidon::new(format!("rc1 {}", i).to_string()).id()),
        );
    }

    let external_idx_1 =
        eval.get_preprocessed_column(Poseidon::new("external_idx_1".to_string()).id());
    let external_idx_2 =
        eval.get_preprocessed_column(Poseidon::new("external_idx_2".to_string()).id());

    let is_external_idx_1_nonzero =
        eval.get_preprocessed_column(Poseidon::new("is_external_idx_1_nonzero".to_string()).id());
    let is_external_idx_2_nonzero =
        eval.get_preprocessed_column(Poseidon::new("is_external_idx_2_nonzero".to_string()).id());

    let swap_bit_addr = rc0[0].clone();

    let in_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());
    let intermediate_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());
    let out_state: [_; N_STATE] = std::array::from_fn(|_| eval.next_trace_mask());

    // if this is first round
    let swap_bit_value = intermediate_state[0].clone();
    let one_minus_swap_bit_value = E::F::one() - swap_bit_value.clone();
    let mut permuted_state: [E::EF; N_STATE] = std::array::from_fn(|i| {
        if i < 8 {
            E::EF::from(in_state[i].clone()) * one_minus_swap_bit_value.clone()
                + E::EF::from(in_state[i + 8].clone()) * swap_bit_value.clone()
        } else {
            E::EF::from(in_state[i - 8].clone()) * swap_bit_value.clone()
                + E::EF::from(in_state[i].clone()) * one_minus_swap_bit_value.clone()
        }
    });
    apply_external_round_matrix(&mut permuted_state);
    (0..N_STATE).for_each(|i| {
        eval.add_constraint(
            (permuted_state[i].clone() - E::EF::from(out_state[i].clone()))
                * is_first_round.clone(),
        );
    });

    // if this is a full2 round
    let mut full_round_state = in_state.clone();
    (0..N_STATE).for_each(|i| {
        full_round_state[i] += rc0[i].clone();
    });
    full_round_state = std::array::from_fn(|i| pow5(full_round_state[i].clone()));
    (0..N_STATE).for_each(|i| {
        eval.add_constraint(
            is_full_round.clone() * (intermediate_state[i].clone() - full_round_state[i].clone()),
        );
        full_round_state[i] = intermediate_state[i].clone();
    });
    apply_external_round_matrix(&mut full_round_state);
    (0..N_STATE).for_each(|i| {
        full_round_state[i] += rc1[i].clone();
    });
    full_round_state = std::array::from_fn(|i| pow5(full_round_state[i].clone()));
    apply_external_round_matrix(&mut full_round_state);
    (0..N_STATE).for_each(|i| {
        eval.add_constraint(
            is_full_round.clone() * (out_state[i].clone() - full_round_state[i].clone()),
        );
    });

    // if this is a partial round
    let mut partial_round_state = in_state.clone();
    for r in 0..N_PARTIAL_ROUNDS {
        partial_round_state[0] += rc0[r].clone();
        partial_round_state[0] = pow5(partial_round_state[0].clone());
        eval.add_constraint(
            is_partial_round.clone()
                * (intermediate_state[r].clone() - partial_round_state[0].clone()),
        );
        partial_round_state[0] = intermediate_state[r].clone();
        apply_internal_round_matrix(&mut partial_round_state);
    }
    (0..N_STATE).for_each(|i| {
        eval.add_constraint(
            is_partial_round.clone() * (out_state[i].clone() - partial_round_state[i].clone()),
        );
    });

    // in_state with id
    let in_left_id = round_id.clone() + round_id.clone();
    let in_right_id = in_left_id.clone() + E::F::one();
    let out_left_id = in_right_id.clone() + E::F::one();
    let out_right_id = out_left_id.clone() + E::F::one();

    let sel = is_external_idx_1_nonzero.clone() * is_first_round.clone();
    let id = is_first_round.clone() * external_idx_1.clone()
        + is_not_first_round.clone() * in_left_id.clone();
    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        -E::EF::from(is_not_first_round.clone()) + sel,
        &[
            id,
            in_state[0].clone(),
            in_state[1].clone(),
            in_state[2].clone(),
            in_state[3].clone(),
            in_state[4].clone(),
            in_state[5].clone(),
            in_state[6].clone(),
            in_state[7].clone(),
        ],
    ));

    let sel = is_external_idx_2_nonzero.clone() * is_first_round.clone();
    let id = is_first_round.clone() * external_idx_2.clone()
        + is_not_first_round.clone() * in_right_id.clone();
    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        -E::EF::from(is_not_first_round.clone()) + sel,
        &[
            id.clone(),
            in_state[8].clone(),
            in_state[9].clone(),
            in_state[10].clone(),
            in_state[11].clone(),
            in_state[12].clone(),
            in_state[13].clone(),
            in_state[14].clone(),
            in_state[15].clone(),
        ],
    ));

    let sel = is_external_idx_1_nonzero.clone() * is_last_round.clone();
    let id = is_last_round.clone() * external_idx_1.clone()
        + is_not_last_round.clone() * out_left_id.clone();
    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::from(is_not_last_round.clone()) + sel.clone(),
        &[
            id.clone(),
            out_state[0].clone(),
            out_state[1].clone(),
            out_state[2].clone(),
            out_state[3].clone(),
            out_state[4].clone(),
            out_state[5].clone(),
            out_state[6].clone(),
            out_state[7].clone(),
        ],
    ));

    let sel = is_external_idx_2_nonzero.clone() * is_last_round.clone();
    let id = is_last_round.clone() * external_idx_2.clone()
        + is_not_last_round.clone() * out_right_id.clone();
    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::from(is_not_last_round.clone()) + sel.clone(),
        &[
            id.clone(),
            out_state[8].clone(),
            out_state[9].clone(),
            out_state[10].clone(),
            out_state[11].clone(),
            out_state[12].clone(),
            out_state[13].clone(),
            out_state[14].clone(),
            out_state[15].clone(),
        ],
    ));

    eval.add_to_relation(RelationEntry::new(
        lookup_elements,
        E::EF::from(is_first_round) * is_not_last_round,
        &[swap_bit_addr, swap_bit_value.clone()],
    ));

    // TODO: use higher degrees batching
    eval.finalize_logup_batched(&vec![0, 0, 0, 1, 1]);
}

#[derive(Clone, Debug)]
pub struct PoseidonEntry {
    pub wire: usize,
    pub hash: [M31; 8],
}

#[derive(Clone, Debug)]
pub struct SwapOption {
    pub addr: usize,
    pub swap: bool,
}

impl Default for SwapOption {
    fn default() -> SwapOption {
        SwapOption {
            addr: 0,
            swap: false,
        }
    }
}

impl SwapOption {
    pub fn one() -> SwapOption {
        SwapOption {
            addr: 1,
            swap: true,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct PoseidonFlow(
    pub  Vec<(
        PoseidonEntry,
        PoseidonEntry,
        PoseidonEntry,
        PoseidonEntry,
        SwapOption,
    )>,
);

pub fn gen_trace(
    flow: &PoseidonFlow,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    // len needs to be padded to 16
    assert_eq!(flow.0.len() % 16, 0);

    // compute the len
    let mut len = flow.0.len() * (N_HALF_FULL_ROUNDS + 1 + 1);
    len = len.next_power_of_two();

    // compute the log size
    let log_size = max(len, N_LANES).ilog2();

    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);

    let mut trace = (0..N_COLUMNS)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    for vec_index in 0..(flow.0.len() / 16) {
        let mut col_index = 0;
        let mut round_index = vec_index * 6;

        // ****** Fill the first round ******
        // Initial state.
        let input_left: [[M31; 8]; N_LANES] =
            std::array::from_fn(|i| flow.0[vec_index * N_LANES + i].0.hash);
        let input_right: [[M31; 8]; N_LANES] =
            std::array::from_fn(|i| flow.0[vec_index * N_LANES + i].1.hash);

        let mut state: [_; N_STATE] = std::array::from_fn(|state_i| {
            if state_i < 8 {
                PackedBaseField::from_array(std::array::from_fn(|i| input_left[i][state_i]))
            } else {
                PackedBaseField::from_array(std::array::from_fn(|i| input_right[i][state_i - 8]))
            }
        });
        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });
        let swap_bit_val = PackedM31::from_array(std::array::from_fn(|i| {
            if flow.0[vec_index * N_LANES + i].4.swap {
                BaseField::one()
            } else {
                BaseField::zero()
            }
        }));
        trace[col_index].data[round_index] = swap_bit_val;
        col_index += 1;
        for _ in 1..N_STATE {
            trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }
        let one_minus_swap_bit_val = PackedM31::one() - swap_bit_val;
        let mut cur = state.clone();
        for i in 0..8 {
            state[i] = cur[i] * one_minus_swap_bit_val + cur[i + 8] * swap_bit_val;
            state[i + 8] = cur[i] * swap_bit_val + cur[i + 8] * one_minus_swap_bit_val;
        }
        apply_external_round_matrix(&mut state);
        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });
        assert_eq!(col_index, N_COLUMNS);

        for r in 0..N_HALF_FULL_ROUNDS / 2 {
            round_index += 1;
            col_index = 0;

            state.iter().copied().for_each(|s| {
                trace[col_index].data[round_index] = s;
                col_index += 1;
            });
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(FIRST_FOUR_ROUND_RC[r * 2][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i]));
            for i in 0..N_STATE {
                trace[col_index].data[round_index] = state[i];
                col_index += 1;
            }
            apply_external_round_matrix(&mut state);
            (0..N_STATE).for_each(|i| {
                state[i] += PackedBaseField::broadcast(FIRST_FOUR_ROUND_RC[r * 2 + 1][i]);
            });
            state = std::array::from_fn(|i| pow5(state[i]));
            apply_external_round_matrix(&mut state);
            state.iter().copied().for_each(|s| {
                trace[col_index].data[round_index] = s;
                col_index += 1;
            });
            assert_eq!(col_index, N_COLUMNS);
        }

        round_index += 1;
        col_index = 0;

        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });

        for r in 0..N_PARTIAL_ROUNDS {
            state[0] += PackedBaseField::broadcast(PARTIAL_ROUNDS_RC[r]);
            state[0] = pow5(state[0]);

            trace[col_index].data[round_index] = state[0];
            col_index += 1;

            apply_internal_round_matrix(&mut state);
        }
        for _ in N_PARTIAL_ROUNDS..N_STATE {
            trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }

        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });

        assert_eq!(col_index, N_COLUMNS);

        // first last round
        round_index += 1;
        col_index = 0;

        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });
        (0..N_STATE).for_each(|i| {
            state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[0][i]);
        });
        state = std::array::from_fn(|i| pow5(state[i]));
        for i in 0..N_STATE {
            trace[col_index].data[round_index] = state[i];
            col_index += 1;
        }
        apply_external_round_matrix(&mut state);
        (0..N_STATE).for_each(|i| {
            state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[1][i]);
        });
        state = std::array::from_fn(|i| pow5(state[i]));
        apply_external_round_matrix(&mut state);
        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });
        assert_eq!(col_index, N_COLUMNS);

        // ****** Fill the last last round ******
        round_index += 1;
        col_index = 0;

        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });
        (0..N_STATE).for_each(|i| {
            state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[2][i]);
        });
        state = std::array::from_fn(|i| pow5(state[i]));
        for i in 0..N_STATE {
            trace[col_index].data[round_index] = state[i];
            col_index += 1;
        }
        apply_external_round_matrix(&mut state);
        (0..N_STATE).for_each(|i| {
            state[i] += PackedBaseField::broadcast(LAST_FOUR_ROUNDS_RC[3][i]);
        });
        state = std::array::from_fn(|i| pow5(state[i]));
        apply_external_round_matrix(&mut state);
        state.iter().copied().for_each(|s| {
            trace[col_index].data[round_index] = s;
            col_index += 1;
        });
        assert_eq!(col_index, N_COLUMNS);
    }

    let padding_start = (flow.0.len() / 16) * 6;
    for round_index in padding_start..(1 << (log_size - LOG_N_LANES)) {
        let mut col_index = 0;
        for _ in 0..N_STATE {
            trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }
        for _ in 0..N_STATE {
            trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }
        for _ in 0..N_STATE {
            trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }
        assert_eq!(col_index, N_COLUMNS);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let trace = trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect();
    trace
}

pub fn gen_constant_trace(
    flow: &PoseidonFlow,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    // len needs to be padded to 16
    assert_eq!(flow.0.len() % 16, 0);

    // compute the len
    let mut len = flow.0.len() * (N_HALF_FULL_ROUNDS + 1 + 1);
    len = len.next_power_of_two();

    // compute the log size
    let log_size = max(len, N_LANES).ilog2();

    let _span = span!(Level::INFO, "Generation").entered();
    assert!(log_size >= LOG_N_LANES);

    let mut constant_trace = (0..40)
        .map(|_| Col::<SimdBackend, BaseField>::zeros(1 << log_size))
        .collect_vec();

    for vec_index in 0..(flow.0.len() / 16) {
        let mut col_index = 0;
        let mut round_index = vec_index * 6;

        // ****** Fill the first round ******
        let is_first_round = PackedM31::one();
        constant_trace[col_index].data[round_index] = is_first_round;
        col_index += 1;
        let is_last_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_last_round;
        col_index += 1;
        let is_full_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_full_round;
        col_index += 1;
        let round_id = PackedM31::from_array(std::array::from_fn(|i| {
            M31::from((vec_index * N_LANES + i) * 6)
        }));
        constant_trace[col_index].data[round_index] = round_id;
        col_index += 1;
        let swap_bit_addr = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(flow.0[vec_index * N_LANES + i].4.addr as u32)
        }));
        constant_trace[col_index].data[round_index] = swap_bit_addr;
        col_index += 1;
        for _ in 1..N_STATE {
            let rc0 = PackedM31::zero();
            constant_trace[col_index].data[round_index] = rc0;
            col_index += 1;
        }
        for _ in 0..N_STATE {
            let rc1 = PackedM31::zero();
            constant_trace[col_index].data[round_index] = rc1;
            col_index += 1;
        }
        let external_idx_1 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(flow.0[vec_index * N_LANES + i].0.wire as u32)
        }));
        constant_trace[col_index].data[round_index] = external_idx_1;
        col_index += 1;
        let external_idx_2 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(flow.0[vec_index * N_LANES + i].1.wire as u32)
        }));
        constant_trace[col_index].data[round_index] = external_idx_2;
        col_index += 1;
        let is_external_idx_1_nonzero = PackedM31::from_array(std::array::from_fn(|i| {
            if flow.0[vec_index * N_LANES + i].0.wire == 0 {
                M31::zero()
            } else {
                M31::one()
            }
        }));
        constant_trace[col_index].data[round_index] = is_external_idx_1_nonzero;
        col_index += 1;
        let is_external_idx_2_nonzero = PackedM31::from_array(std::array::from_fn(|i| {
            if flow.0[vec_index * N_LANES + i].1.wire == 0 {
                M31::zero()
            } else {
                M31::one()
            }
        }));
        constant_trace[col_index].data[round_index] = is_external_idx_2_nonzero;
        col_index += 1;
        assert_eq!(col_index, 8 + 16 + 16);

        for r in 0..N_HALF_FULL_ROUNDS / 2 {
            round_index += 1;
            col_index = 0;

            let is_first_round = PackedM31::zero();
            constant_trace[col_index].data[round_index] = is_first_round;
            col_index += 1;
            let is_last_round = PackedM31::zero();
            constant_trace[col_index].data[round_index] = is_last_round;
            col_index += 1;
            let is_full_round = PackedM31::one();
            constant_trace[col_index].data[round_index] = is_full_round;
            col_index += 1;

            let round_id = PackedM31::from_array(std::array::from_fn(|i| {
                M31::from((vec_index * N_LANES + i) * 6 + 1 + r)
            }));
            constant_trace[col_index].data[round_index] = round_id;
            col_index += 1;
            for i in 0..N_STATE {
                let rc0 = PackedM31::broadcast(FIRST_FOUR_ROUND_RC[r * 2][i]);
                constant_trace[col_index].data[round_index] = rc0;
                col_index += 1;
            }
            for i in 0..N_STATE {
                let rc1 = PackedM31::broadcast(FIRST_FOUR_ROUND_RC[r * 2 + 1][i]);
                constant_trace[col_index].data[round_index] = rc1;
                col_index += 1;
            }
            constant_trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
            constant_trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
            constant_trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
            constant_trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
            assert_eq!(col_index, 8 + 16 + 16);
        }

        round_index += 1;
        col_index = 0;

        let is_first_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_first_round;
        col_index += 1;
        let is_last_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_last_round;
        col_index += 1;
        let is_full_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_full_round;
        col_index += 1;

        let round_id = PackedM31::from_array(std::array::from_fn(|i| {
            M31::from((vec_index * N_LANES + i) * 6 + 1 + 2)
        }));
        constant_trace[col_index].data[round_index] = round_id;
        col_index += 1;

        for r in 0..N_PARTIAL_ROUNDS {
            let rc0 = PackedM31::broadcast(PARTIAL_ROUNDS_RC[r]);
            constant_trace[col_index].data[round_index] = rc0;
            col_index += 1;
        }
        for _ in N_PARTIAL_ROUNDS..N_STATE {
            let rc0 = PackedM31::zero();
            constant_trace[col_index].data[round_index] = rc0;
            col_index += 1;
        }
        for _ in 0..N_STATE {
            let rc1 = PackedM31::zero();
            constant_trace[col_index].data[round_index] = rc1;
            col_index += 1;
        }

        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        assert_eq!(col_index, 8 + 16 + 16);

        round_index += 1;
        col_index = 0;
        let is_first_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_first_round;
        col_index += 1;
        let is_last_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_last_round;
        col_index += 1;
        let is_full_round = PackedM31::one();
        constant_trace[col_index].data[round_index] = is_full_round;
        col_index += 1;

        let round_id = PackedM31::from_array(std::array::from_fn(|i| {
            M31::from((vec_index * N_LANES + i) * 6 + 1 + 2 + 1)
        }));
        constant_trace[col_index].data[round_index] = round_id;
        col_index += 1;
        for i in 0..N_STATE {
            let rc0 = PackedM31::broadcast(LAST_FOUR_ROUNDS_RC[0][i]);
            constant_trace[col_index].data[round_index] = rc0;
            col_index += 1;
        }
        for i in 0..N_STATE {
            let rc1 = PackedM31::broadcast(LAST_FOUR_ROUNDS_RC[1][i]);
            constant_trace[col_index].data[round_index] = rc1;
            col_index += 1;
        }
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        assert_eq!(col_index, 8 + 16 + 16);

        round_index += 1;
        col_index = 0;

        let is_first_round = PackedM31::zero();
        constant_trace[col_index].data[round_index] = is_first_round;
        col_index += 1;
        let is_last_round = PackedM31::one();
        constant_trace[col_index].data[round_index] = is_last_round;
        col_index += 1;
        let is_full_round = PackedM31::one();
        constant_trace[col_index].data[round_index] = is_full_round;
        col_index += 1;

        let round_id = PackedM31::from_array(std::array::from_fn(|i| {
            M31::from((vec_index * N_LANES + i) * 6 + 1 + 2 + 1 + 1)
        }));
        constant_trace[col_index].data[round_index] = round_id;
        col_index += 1;
        for i in 0..N_STATE {
            let rc0 = PackedM31::broadcast(LAST_FOUR_ROUNDS_RC[2][i]);
            constant_trace[col_index].data[round_index] = rc0;
            col_index += 1;
        }
        for i in 0..N_STATE {
            let rc1 = PackedM31::broadcast(LAST_FOUR_ROUNDS_RC[3][i]);
            constant_trace[col_index].data[round_index] = rc1;
            col_index += 1;
        }
        let external_idx_1 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(flow.0[vec_index * N_LANES + i].2.wire as u32)
        }));
        constant_trace[col_index].data[round_index] = external_idx_1;
        col_index += 1;
        let external_idx_2 = PackedM31::from_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(flow.0[vec_index * N_LANES + i].3.wire as u32)
        }));
        constant_trace[col_index].data[round_index] = external_idx_2;
        col_index += 1;
        let is_external_idx_1_nonzero = PackedM31::from_array(std::array::from_fn(|i| {
            if flow.0[vec_index * N_LANES + i].2.wire == 0 {
                M31::zero()
            } else {
                M31::one()
            }
        }));
        constant_trace[col_index].data[round_index] = is_external_idx_1_nonzero;
        col_index += 1;
        let is_external_idx_2_nonzero = PackedM31::from_array(std::array::from_fn(|i| {
            if flow.0[vec_index * N_LANES + i].3.wire == 0 {
                M31::zero()
            } else {
                M31::one()
            }
        }));
        constant_trace[col_index].data[round_index] = is_external_idx_2_nonzero;
        col_index += 1;
        assert_eq!(col_index, 8 + 16 + 16);
    }

    let padding_start = (flow.0.len() / 16) * 6;
    for round_index in padding_start..(1 << (log_size - LOG_N_LANES)) {
        let mut col_index = 0;

        constant_trace[col_index].data[round_index] = PackedM31::one();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::one();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;

        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;

        for _ in 0..N_STATE {
            constant_trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }
        for _ in 0..N_STATE {
            constant_trace[col_index].data[round_index] = PackedM31::zero();
            col_index += 1;
        }

        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;
        constant_trace[col_index].data[round_index] = PackedM31::zero();
        col_index += 1;

        assert_eq!(col_index, 8 + 16 + 16);
    }

    let domain = CanonicCoset::new(log_size).circle_domain();
    let mut trace = constant_trace
        .into_iter()
        .map(|eval| CircleEvaluation::new(domain, eval))
        .collect_vec();
    trace
}

pub fn check_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    constant_trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    l: usize,
) {
    assert_eq!(trace.len(), N_COLUMNS);

    let log_size = trace[0].length.ilog2();

    fn apply_4x4_mds_matrix(
        x0: PackedM31,
        x1: PackedM31,
        x2: PackedM31,
        x3: PackedM31,
    ) -> [PackedM31; 4] {
        let t0 = x0 + x1;
        let t1 = x2 + x3;
        let t2 = x1.double() + t1;
        let t3 = x3.double() + t0;
        let t4 = t1.double().double() + t3;
        let t5 = t0.double().double() + t2;
        let t6 = t3 + t5;
        let t7 = t2 + t4;

        [t6, t5, t7, t4]
    }

    fn apply_16x16_mds_matrix(state: &mut [PackedM31; 16]) {
        let p1 = apply_4x4_mds_matrix(state[0], state[1], state[2], state[3]);
        let p2 = apply_4x4_mds_matrix(state[4], state[5], state[6], state[7]);
        let p3 = apply_4x4_mds_matrix(state[8], state[9], state[10], state[11]);
        let p4 = apply_4x4_mds_matrix(state[12], state[13], state[14], state[15]);

        let t = [
            p1[0] + p2[0] + p3[0] + p4[0],
            p1[1] + p2[1] + p3[1] + p4[1],
            p1[2] + p2[2] + p3[2] + p4[2],
            p1[3] + p2[3] + p3[3] + p4[3],
        ];

        for i in 0..4 {
            state[i] = p1[i] + t[i];
            state[i + 4] = p2[i] + t[i];
            state[i + 8] = p3[i] + t[i];
            state[i + 12] = p4[i] + t[i];
        }
    }

    fn pow5(v: PackedM31) -> PackedM31 {
        let t = v * v;
        t * t * v
    }

    assert_eq!(l % N_LANES, 0);
    let padding_start = l / N_LANES * 6;

    for vec_index in 0..1 << (log_size - LOG_N_LANES) {
        let mut constant_col_index = 0;
        let mut trace_col_index = 0;

        let is_first_round = constant_trace[constant_col_index].data[vec_index];
        constant_col_index += 1;
        let is_not_first_round = PackedM31::one() - is_first_round;

        let is_last_round = constant_trace[constant_col_index].data[vec_index];
        constant_col_index += 1;

        let in_state: [_; N_STATE] =
            std::array::from_fn(|i| trace[trace_col_index + i].data[vec_index]);
        trace_col_index += N_STATE;

        let intermediate_state: [_; N_STATE] =
            std::array::from_fn(|i| trace[trace_col_index + i].data[vec_index]);
        trace_col_index += N_STATE;

        let swap_addr_val = intermediate_state[0];
        let one_minus_swap_addr_val = PackedM31::one() - swap_addr_val;

        let out_state: [_; N_STATE] =
            std::array::from_fn(|i| trace[trace_col_index + i].data[vec_index]);
        trace_col_index += N_STATE;

        let mut permuted_state = in_state.clone();
        for i in 0..8 {
            permuted_state[i] =
                in_state[i] * one_minus_swap_addr_val + in_state[i + 8] * swap_addr_val;
            permuted_state[i + 8] =
                in_state[i] * swap_addr_val + in_state[i + 8] * one_minus_swap_addr_val;
        }
        apply_external_round_matrix(&mut permuted_state);
        (0..N_STATE).for_each(|i| {
            assert!(
                (is_first_round.clone() * (out_state[i].clone() - permuted_state[i].clone()))
                    .is_zero()
            );
        });

        let is_full_round = constant_trace[constant_col_index].data[vec_index];
        constant_col_index += 1;

        let is_partial_round = is_not_first_round - is_full_round;

        let round_id = constant_trace[constant_col_index].data[vec_index];
        constant_col_index += 1;

        let rc0: [_; N_STATE] =
            std::array::from_fn(|i| constant_trace[constant_col_index + i].data[vec_index]);
        constant_col_index += N_STATE;
        let rc1: [_; N_STATE] =
            std::array::from_fn(|i| constant_trace[constant_col_index + i].data[vec_index]);
        constant_col_index += N_STATE;

        let mut full_round_state = in_state.clone();
        (0..N_STATE).for_each(|i| {
            full_round_state[i] += rc0[i];
        });
        full_round_state = std::array::from_fn(|i| pow5(full_round_state[i].clone()));
        (0..N_STATE).for_each(|i| {
            assert!((is_full_round.clone()
                * (intermediate_state[i].clone() - full_round_state[i].clone()))
            .is_zero());
        });
        apply_external_round_matrix(&mut full_round_state);
        (0..N_STATE).for_each(|i| {
            full_round_state[i] += rc1[i];
        });
        full_round_state = std::array::from_fn(|i| pow5(full_round_state[i].clone()));
        apply_external_round_matrix(&mut full_round_state);
        (0..N_STATE).for_each(|i| {
            assert!(
                (is_full_round.clone() * (out_state[i].clone() - full_round_state[i].clone()))
                    .is_zero()
            );
        });

        let mut partial_round_state = in_state.clone();
        for r in 0..N_PARTIAL_ROUNDS {
            partial_round_state[0] += rc0[r].clone();
            partial_round_state[0] = pow5(partial_round_state[0].clone());
            assert!((is_partial_round.clone()
                * (intermediate_state[r].clone() - partial_round_state[0].clone()))
            .is_zero());
            apply_internal_round_matrix(&mut partial_round_state);
        }
        (0..N_STATE).for_each(|i| {
            assert!((is_partial_round.clone()
                * (out_state[i].clone() - partial_round_state[i].clone()))
            .is_zero());
        });
        assert_eq!(trace_col_index, N_COLUMNS);
    }
}

pub fn gen_interaction_trace(
    trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    constant_trace: &ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    lookup_elements: &PlonkWithAcceleratorLookupElements,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let len = trace[0].length;

    // compute the log size
    let log_size = len.ilog2();

    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let is_first_round = constant_trace[0].data[vec_row];
        let is_last_round = constant_trace[1].data[vec_row];
        let round_id = constant_trace[3].data[vec_row];

        let is_not_first_round = PackedBaseField::one() - is_first_round;
        let is_not_last_round = PackedBaseField::one() - is_last_round;

        let external_idx_1 = constant_trace[36].data[vec_row];
        let external_idx_2 = constant_trace[37].data[vec_row];
        let is_external_idx_1_nonzero = constant_trace[38].data[vec_row];
        let is_external_idx_2_nonzero = constant_trace[39].data[vec_row];
        let swap_bit_addr = constant_trace[4].data[vec_row];

        let in_left_id = round_id + round_id;
        let in_right_id = in_left_id + M31::one();
        let out_left_id = in_right_id + M31::one();
        let out_right_id = out_left_id + M31::one();

        let sel = is_external_idx_1_nonzero * is_first_round.clone();
        let id = is_first_round * external_idx_1 + is_not_first_round * in_left_id;
        let num0 = sel - is_not_first_round;
        let mut denom0_arr = Vec::with_capacity(9);
        denom0_arr.push(id);
        for j in 0..8 {
            denom0_arr.push(trace[j].data[vec_row]);
        }
        let denom0: PackedSecureField = lookup_elements.combine(&denom0_arr);

        let sel = is_external_idx_2_nonzero * is_first_round.clone();
        let id = is_first_round * external_idx_2 + is_not_first_round * in_right_id;
        let num1 = sel - is_not_first_round;
        let mut denom1_arr = Vec::with_capacity(9);
        denom1_arr.push(id);
        for j in 0..8 {
            denom1_arr.push(trace[8 + j].data[vec_row]);
        }
        let denom1: PackedSecureField = lookup_elements.combine(&denom1_arr);

        let mut part1_denom = denom0 * denom1;
        let mut part1_num = denom1 * num0 + denom0 * num1;

        let sel = is_external_idx_1_nonzero.clone() * is_last_round.clone();
        let id = is_last_round * external_idx_1 + is_not_last_round * out_left_id;
        let num0 = sel + is_not_last_round;
        let mut denom0_arr = Vec::with_capacity(9);
        denom0_arr.push(id);
        for j in 0..8 {
            denom0_arr.push(trace[32 + j].data[vec_row]);
        }
        let denom0: PackedSecureField = lookup_elements.combine(&denom0_arr);

        col_gen.write_frac(
            vec_row,
            part1_denom * num0 + denom0 * part1_num,
            part1_denom * denom0,
        );
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let is_first_round = constant_trace[0].data[vec_row];
        let is_last_round = constant_trace[1].data[vec_row];
        let round_id = constant_trace[3].data[vec_row];
        let is_not_last_round = PackedBaseField::one() - is_last_round;

        let external_idx_1 = constant_trace[36].data[vec_row];
        let external_idx_2 = constant_trace[37].data[vec_row];
        let is_external_idx_1_nonzero = constant_trace[38].data[vec_row];
        let is_external_idx_2_nonzero = constant_trace[39].data[vec_row];
        let swap_bit_addr = constant_trace[4].data[vec_row];

        let in_left_id = round_id + round_id;
        let in_right_id = in_left_id + M31::one();
        let out_left_id = in_right_id + M31::one();
        let out_right_id = out_left_id + M31::one();

        let sel = is_external_idx_2_nonzero.clone() * is_last_round.clone();
        let id = is_last_round * external_idx_2 + is_not_last_round * out_right_id;
        let num0 = sel + is_not_last_round;
        let mut denom0_arr = Vec::with_capacity(9);
        denom0_arr.push(id);
        for j in 0..8 {
            denom0_arr.push(trace[40 + j].data[vec_row]);
        }
        let denom0: PackedSecureField = lookup_elements.combine(&denom0_arr);

        let swap_bit_val = trace[16].data[vec_row];
        let num1 = is_first_round * is_not_last_round;
        let denom1: PackedSecureField = lookup_elements.combine(&[swap_bit_addr, swap_bit_val]);

        col_gen.write_frac(vec_row, denom1 * num0 + denom0 * num1, denom1 * denom0);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last()
}

pub fn prove_poseidon_accelerator<MC: MerkleChannel>(
    config: PcsConfig,
    flow: &mut PoseidonFlow,
) -> (PoseidonAcceleratorComponent, StarkProof<MC::H>)
where
    SimdBackend: BackendForChannel<MC>,
{
    // compute the len
    let mut len = flow.0.len() * (N_HALF_FULL_ROUNDS + 1 + 1);
    len = len.next_power_of_two();

    // compute the log size
    let log_n_rows = max(len, N_LANES).ilog2();

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + LOG_EXPAND)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::<_, MC>::new(config, &twiddles);

    let trace = gen_trace(flow);
    let constant_trace = gen_constant_trace(flow);
    check_trace(&trace, &constant_trace, flow.0.len());

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(constant_trace.clone());
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace.clone());
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (interaction_trace, total_sum) =
        gen_interaction_trace(&trace, &constant_trace, &lookup_elements);
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace);
    tree_builder.commit(channel);
    span.exit();

    // Prove constraints.
    let component = PoseidonAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PoseidonAcceleratorEval {
            log_n_rows,
            lookup_elements,
            total_sum,
        },
        total_sum,
    );

    // Sanity check. Remove for production.
    let trace_polys = commitment_scheme
        .trees
        .as_ref()
        .map(|t| t.polynomials.iter().cloned().collect_vec());
    assert_constraints(
        &trace_polys,
        CanonicCoset::new(log_n_rows),
        |eval| {
            component.evaluate(eval);
        },
        total_sum,
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
}

#[allow(unused)]
pub fn prove_test_poseidon_accelerator(
    log_n_instances: u32,
    config: PcsConfig,
) -> (
    PoseidonAcceleratorComponent,
    StarkProof<Blake2sMerkleHasher>,
) {
    // Additional test constants
    const TEST_1: [BaseField; 8] = [
        BaseField::from_u32_unchecked(0),
        BaseField::from_u32_unchecked(1),
        BaseField::from_u32_unchecked(2),
        BaseField::from_u32_unchecked(3),
        BaseField::from_u32_unchecked(4),
        BaseField::from_u32_unchecked(5),
        BaseField::from_u32_unchecked(6),
        BaseField::from_u32_unchecked(7),
    ];
    const TEST_2: [BaseField; 8] = [
        BaseField::from_u32_unchecked(8),
        BaseField::from_u32_unchecked(9),
        BaseField::from_u32_unchecked(10),
        BaseField::from_u32_unchecked(11),
        BaseField::from_u32_unchecked(12),
        BaseField::from_u32_unchecked(13),
        BaseField::from_u32_unchecked(14),
        BaseField::from_u32_unchecked(15),
    ];

    const TEST_3: [BaseField; 8] = [
        BaseField::from_u32_unchecked(0x0f8b2223),
        BaseField::from_u32_unchecked(0x4681926c),
        BaseField::from_u32_unchecked(0x62bf39da),
        BaseField::from_u32_unchecked(0x2c775855),
        BaseField::from_u32_unchecked(0x0621c328),
        BaseField::from_u32_unchecked(0x6c092e66),
        BaseField::from_u32_unchecked(0x1ebf9d29),
        BaseField::from_u32_unchecked(0x2d015c8e),
    ];

    const TEST_4: [BaseField; 8] = [
        BaseField::from_u32_unchecked(0x6447f97c),
        BaseField::from_u32_unchecked(0x4b6cc939),
        BaseField::from_u32_unchecked(0x0e395f63),
        BaseField::from_u32_unchecked(0x0bf7f688),
        BaseField::from_u32_unchecked(0x62ed4765),
        BaseField::from_u32_unchecked(0x7bfd5e1b),
        BaseField::from_u32_unchecked(0x4bafb4b0),
        BaseField::from_u32_unchecked(0x4cc30530),
    ];

    // Prepare a fibonacci circuit.
    assert!(log_n_instances >= LOG_N_LANES);
    let n_instances = (1 << log_n_instances) as usize;

    let mut flow = PoseidonFlow::default();

    let mut prng = SmallRng::seed_from_u64(0);
    for _ in 0..n_instances {
        if prng.gen::<bool>() == true {
            flow.0.push((
                PoseidonEntry {
                    wire: 123001,
                    hash: CONSTANT_1,
                },
                PoseidonEntry {
                    wire: 123001,
                    hash: CONSTANT_1,
                },
                PoseidonEntry {
                    wire: 123002,
                    hash: CONSTANT_2,
                },
                PoseidonEntry {
                    wire: 123003,
                    hash: CONSTANT_3,
                },
                SwapOption::default(),
            ));
        } else {
            flow.0.push((
                PoseidonEntry {
                    wire: 256002,
                    hash: TEST_2,
                },
                PoseidonEntry {
                    wire: 256001,
                    hash: TEST_1,
                },
                PoseidonEntry {
                    wire: 256003,
                    hash: TEST_3,
                },
                PoseidonEntry {
                    wire: 256004,
                    hash: TEST_4,
                },
                SwapOption::one(),
            ));
        }
    }

    prove_poseidon_accelerator::<Blake2sMerkleChannel>(config, &mut flow)
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::core::air::Component;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fri::FriConfig;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::examples::plonk_with_poseidon::plonk::PlonkWithAcceleratorLookupElements;
    use crate::examples::plonk_with_poseidon::poseidon::prove_test_poseidon_accelerator;

    #[test_log::test]
    fn test_simd_poseidon_prove() {
        // Get from environment variable:
        let log_n_instances = env::var("LOG_N_INSTANCES")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<u32>()
            .unwrap();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(5, 4, 64),
        };

        // Prove.
        let (component, proof) = prove_test_poseidon_accelerator(log_n_instances, config);

        // Verify.
        // TODO: Create Air instance independently.
        let channel = &mut Blake2sChannel::default();
        let commitment_scheme = &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

        // Decommit.
        // Retrieve the expected column sizes in each commitment interaction, from the AIR.
        let sizes = component.trace_log_degree_bounds();

        // Preprocessed columns.
        commitment_scheme.commit(proof.commitments[0], &sizes[0], channel);
        // Trace columns.
        commitment_scheme.commit(proof.commitments[1], &sizes[1], channel);
        // Draw lookup element.
        let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);
        assert_eq!(lookup_elements, component.lookup_elements);
        // Interaction columns.
        commitment_scheme.commit(proof.commitments[2], &sizes[2], channel);

        verify(&[&component], channel, commitment_scheme, proof).unwrap();
    }
}
