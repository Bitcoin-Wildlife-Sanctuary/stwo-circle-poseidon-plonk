use std::cmp::max;

use itertools::{chain, Itertools};
use num_traits::{One, Zero};
use serde::{Deserialize, Serialize};
use tracing::{span, Level};

use crate::constraint_framework::preprocessed_columns::{gen_is_first, PreprocessedColumn};
use crate::constraint_framework::{
    Relation, TraceLocationAllocator, INTERACTION_TRACE_IDX, ORIGINAL_TRACE_IDX,
    PREPROCESSED_TRACE_IDX,
};
use crate::core::air::{Component, ComponentProver};
use crate::core::backend::simd::m31::LOG_N_LANES;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::BackendForChannel;
use crate::core::channel::{Channel, MerkleChannel};
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::SecureField;
use crate::core::fields::FieldExpOps;
use crate::core::pcs::{CommitmentSchemeProver, CommitmentSchemeVerifier, PcsConfig, TreeVec};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, verify, StarkProof, VerificationError};
use crate::core::vcs::ops::MerkleHasher;
use crate::examples::plonk_with_poseidon::plonk::{
    PlonkWithAcceleratorCircuitTrace, PlonkWithAcceleratorComponent, PlonkWithAcceleratorEval,
    PlonkWithAcceleratorLookupElements,
};
use crate::examples::plonk_with_poseidon::poseidon::{
    check_interaction_trace, check_trace, PoseidonAcceleratorComponent, PoseidonAcceleratorEval,
    PoseidonMetadata,
};
use crate::examples::plonk_with_poseidon::{plonk, poseidon};

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithPoseidonStatement0 {
    pub log_size_plonk: u32,
    pub log_size_poseidon: u32,
}

impl PlonkWithPoseidonStatement0 {
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut sizes = TreeVec::new(vec![vec![], vec![], vec![]]);

        let log_size_plonk = self.log_size_plonk;
        let log_size_poseidon = self.log_size_poseidon;

        sizes[PREPROCESSED_TRACE_IDX].extend_from_slice(&[log_size_plonk; 7]);
        sizes[PREPROCESSED_TRACE_IDX].extend_from_slice(&[log_size_poseidon; 5]);

        sizes[ORIGINAL_TRACE_IDX].extend_from_slice(&[log_size_plonk; 3]);
        sizes[ORIGINAL_TRACE_IDX].extend_from_slice(&[log_size_poseidon; 162]);

        sizes[INTERACTION_TRACE_IDX].extend_from_slice(&[log_size_plonk; 8]);
        sizes[INTERACTION_TRACE_IDX].extend_from_slice(&[log_size_poseidon; 16]);

        sizes
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size_plonk as u64);
        channel.mix_u64(self.log_size_poseidon as u64);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithPoseidonStatement1 {
    pub plonk_total_sum: SecureField,
    pub poseidon_total_sum: SecureField,
}

impl PlonkWithPoseidonStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.plonk_total_sum, self.poseidon_total_sum]);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithPoseidonProof<H: MerkleHasher> {
    pub stmt0: PlonkWithPoseidonStatement0,
    pub stmt1: PlonkWithPoseidonStatement1,
    pub stark_proof: StarkProof<H>,
}

pub struct PlonkWithPoseidonComponents {
    pub plonk: PlonkWithAcceleratorComponent,
    pub poseidon: PoseidonAcceleratorComponent,
}

impl PlonkWithPoseidonComponents {
    pub fn new(
        stmt0: &PlonkWithPoseidonStatement0,
        lookup_elements: &PlonkWithAcceleratorLookupElements,
        stmt1: &PlonkWithPoseidonStatement1,
    ) -> Self {
        let tree_span_provider = &mut TraceLocationAllocator::new_with_preproccessed_columns(
            &chain!(
                [
                    PreprocessedColumn::Plonk(0),
                    PreprocessedColumn::Plonk(1),
                    PreprocessedColumn::Plonk(2),
                    PreprocessedColumn::Plonk(3),
                    PreprocessedColumn::Plonk(4),
                    PreprocessedColumn::Plonk(5),
                    PreprocessedColumn::IsFirst(stmt0.log_size_plonk as u32),
                ],
                [
                    PreprocessedColumn::Poseidon(0),
                    PreprocessedColumn::Poseidon(1),
                    PreprocessedColumn::Poseidon(2),
                    PreprocessedColumn::Poseidon(3),
                    PreprocessedColumn::IsFirst(stmt0.log_size_poseidon as u32),
                ]
            )
            .collect_vec()[..],
        );

        Self {
            plonk: PlonkWithAcceleratorComponent::new(
                tree_span_provider,
                PlonkWithAcceleratorEval {
                    log_n_rows: stmt0.log_size_plonk as u32,
                    lookup_elements: lookup_elements.clone(),
                    total_sum: stmt1.plonk_total_sum,
                },
                (stmt1.plonk_total_sum, None),
            ),
            poseidon: PoseidonAcceleratorComponent::new(
                tree_span_provider,
                PoseidonAcceleratorEval {
                    log_n_rows: stmt0.log_size_poseidon as u32,
                    lookup_elements: lookup_elements.clone(),
                    total_sum: stmt1.poseidon_total_sum,
                },
                (stmt1.poseidon_total_sum, None),
            ),
        }
    }

    pub fn components(&self) -> Vec<&dyn Component> {
        vec![
            &self.plonk as &dyn Component,
            &self.poseidon as &dyn Component,
        ]
    }

    pub fn component_provers(&self) -> Vec<&dyn ComponentProver<SimdBackend>> {
        vec![
            &self.plonk as &dyn ComponentProver<SimdBackend>,
            &self.poseidon as &dyn ComponentProver<SimdBackend>,
        ]
    }
}

pub fn prove_plonk_with_poseidon<MC: MerkleChannel>(
    log_size_plonk: u32,
    log_size_poseidon: u32,
    config: PcsConfig,
    circuit: &PlonkWithAcceleratorCircuitTrace,
    metadata: &mut PoseidonMetadata,
) -> PlonkWithPoseidonProof<MC::H>
where
    SimdBackend: BackendForChannel<MC>,
{
    assert!(log_size_plonk >= LOG_N_LANES);
    assert!(log_size_poseidon >= LOG_N_LANES);
    assert_eq!(circuit.mult.length, 1 << log_size_plonk);
    assert_eq!(
        metadata.prescribed_flow.addr_1.len(),
        1 << log_size_poseidon
    );

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let log_max_rows = max(log_size_plonk + 1, log_size_poseidon + 2);
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_max_rows + config.fri_config.log_blowup_factor)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut MC::C::default();
    let mut commitment_scheme = CommitmentSchemeProver::new(config, &twiddles);

    // Preprocessed trace
    let is_first = gen_is_first(log_size_plonk);
    let mut plonk_constant_trace = [
        circuit.a_wire.clone(),
        circuit.b_wire.clone(),
        circuit.c_wire.clone(),
        circuit.op.clone(),
        circuit.mult.clone(),
        circuit.mult_poseidon.clone(),
    ]
    .into_iter()
    .map(|col| {
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new_canonical_ordered(
            CanonicCoset::new(log_size_plonk),
            col,
        )
    })
    .collect_vec();
    plonk_constant_trace.push(is_first);

    let poseidon_trace = poseidon::gen_trace(metadata);
    check_trace(&poseidon_trace);
    let poseidon_constant_trace = poseidon::gen_constant_trace(metadata);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![
        plonk_constant_trace,
        poseidon_constant_trace.clone(),
    ]);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let plonk_trace = plonk::gen_trace(log_size_plonk, &circuit);

    // Statement0.
    let stmt0 = PlonkWithPoseidonStatement0 {
        log_size_plonk,
        log_size_poseidon,
    };
    stmt0.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![plonk_trace, poseidon_trace.clone()]);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (plonk_interaction_trace, plonk_total_sum) =
        plonk::gen_interaction_trace(log_size_plonk, &circuit, &lookup_elements);
    let (poseidon_interaction_trace, poseidon_total_sum) =
        poseidon::gen_interaction_trace(metadata, &lookup_elements);
    check_interaction_trace(
        &poseidon_trace,
        &poseidon_interaction_trace,
        &poseidon_constant_trace,
        &lookup_elements,
        poseidon_total_sum,
    );
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(chain![plonk_interaction_trace, poseidon_interaction_trace]);
    // Statement1.
    let stmt1 = PlonkWithPoseidonStatement1 {
        plonk_total_sum,
        poseidon_total_sum,
    };
    stmt1.mix_into(channel);
    tree_builder.commit(channel);
    span.exit();

    assert_eq!(
        commitment_scheme
            .polynomials()
            .as_cols_ref()
            .map_cols(|c| c.log_size())
            .0,
        stmt0.log_sizes().0
    );

    // Prove constraints.
    let components = PlonkWithPoseidonComponents::new(&stmt0, &lookup_elements, &stmt1);
    let stark_proof = prove(&components.component_provers(), channel, commitment_scheme).unwrap();

    PlonkWithPoseidonProof {
        stmt0,
        stmt1,
        stark_proof,
    }
}

#[allow(unused)]
pub fn verify_plonk_with_poseidon<MC: MerkleChannel>(
    PlonkWithPoseidonProof {
        stmt0,
        stmt1,
        stark_proof,
    }: PlonkWithPoseidonProof<MC::H>,
    config: PcsConfig,
) -> Result<(), VerificationError> {
    let channel = &mut MC::C::default();
    let commitment_scheme = &mut CommitmentSchemeVerifier::<MC>::new(config);

    let log_sizes = stmt0.log_sizes();

    // Preprocessed trace.
    commitment_scheme.commit(stark_proof.commitments[0], &log_sizes[0], channel);

    // Trace.
    stmt0.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[1], &log_sizes[1], channel);

    // Draw interaction elements.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    stmt1.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

    let components = PlonkWithPoseidonComponents::new(&stmt0, &lookup_elements, &stmt1);
    let one_sum: SecureField = lookup_elements.combine(&[M31::one(), M31::one()]);

    let total_sum = stmt1.plonk_total_sum - one_sum.inverse() + stmt1.poseidon_total_sum;
    assert_eq!(total_sum, SecureField::zero());

    verify(
        &components.components(),
        channel,
        commitment_scheme,
        stark_proof,
    )
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use num_traits::{One, Zero};

    use crate::core::air::Component;
    use crate::core::channel::Blake2sChannel;
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fri::FriConfig;
    use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig};
    use crate::core::prover::verify;
    use crate::core::vcs::blake2_merkle::Blake2sMerkleChannel;
    use crate::core::vcs::poseidon31_merkle::{Poseidon31MerkleChannel, Poseidon31MerkleHasher};
    use crate::examples::plonk_with_poseidon::air::{
        prove_plonk_with_poseidon, verify_plonk_with_poseidon, PlonkWithPoseidonProof,
    };
    use crate::examples::plonk_with_poseidon::plonk::{
        prove_plonk_with_accelerator, PlonkWithAcceleratorCircuitTrace,
        PlonkWithAcceleratorLookupElements,
    };
    use crate::examples::plonk_with_poseidon::poseidon::{
        prove_poseidon_accelerator, PoseidonControlFlow, PoseidonDataFlow, PoseidonMetadata,
        PoseidonPrescribedFlow, CONSTANT_1, CONSTANT_2, CONSTANT_3,
    };

    fn generate_test_circuit() -> (PlonkWithAcceleratorCircuitTrace, PoseidonMetadata) {
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

        let mut mult = vec![];
        let mut mult_poseidon = vec![];
        let mut a_wire = vec![];
        let mut b_wire = vec![];
        let mut c_wire = vec![];
        let mut op = vec![];
        let mut variables = vec![];

        // allocate 0
        mult.push(3);
        mult_poseidon.push(0);
        a_wire.push(0);
        b_wire.push(0);
        c_wire.push(0);
        op.push(M31::one());
        variables.push(M31::zero());

        // allocate 1
        mult.push(0); // intentionally reduced by 1 to request for an input
        mult_poseidon.push(0);
        a_wire.push(1);
        b_wire.push(0);
        op.push(M31::one());
        variables.push(M31::one());

        let mut cur_idx = 2usize;
        let mut hash_idx = vec![];

        // allocate TEST_1 to TEST_4
        for g in [TEST_1, TEST_2, TEST_3, TEST_4] {
            for i in 0..8 {
                mult.push(1);
                mult_poseidon.push(0);
                if i == 0 {
                    hash_idx.push(cur_idx);
                }
                a_wire.push(cur_idx);
                b_wire.push(1);
                op.push(M31::zero());
                variables.push(g[i]);

                mult[1] += 1;
                cur_idx += 1;
            }
        }

        assert_eq!(hash_idx.len(), 4);

        // allocate the hash_idx addresses
        let mut addr_hash_idx = vec![];
        for idx in hash_idx.iter() {
            mult.push(1);
            mult_poseidon.push(0);
            a_wire.push(cur_idx);
            b_wire.push(1);
            op.push(M31::zero());
            variables.push(M31::from_u32_unchecked(*idx as u32));

            addr_hash_idx.push(cur_idx);
            mult[1] += 1;
            cur_idx += 1;
        }

        let mut constant_idx = vec![];
        // allocate CONSTANT_1 to CONSTANT_3
        for g in [CONSTANT_1, CONSTANT_2, CONSTANT_3] {
            for i in 0..8 {
                mult.push(1);
                mult_poseidon.push(0);
                if i == 0 {
                    constant_idx.push(cur_idx);
                }
                a_wire.push(cur_idx);
                b_wire.push(1);
                op.push(M31::zero());
                variables.push(g[i]);

                mult[1] += 1;
                cur_idx += 1;
            }
        }

        // allocate the constant_idx addresses
        let mut addr_constant_idx = vec![];
        for idx in constant_idx.iter() {
            mult.push(1);
            mult_poseidon.push(0);
            a_wire.push(cur_idx);
            b_wire.push(1);
            op.push(M31::zero());
            variables.push(M31::from_u32_unchecked(*idx as u32));

            addr_constant_idx.push(cur_idx);
            mult[1] += 1;
            cur_idx += 1;
        }

        // assume that Poseidon has 32 gates,
        // 16 of them will be dealing with CONSTANT_1, _2, _3
        // 16 of them will be dealing with TEST_1, _2, _3, _4
        mult[addr_hash_idx[0]] += 16;
        mult[addr_hash_idx[1]] += 16;
        mult[addr_hash_idx[2]] += 16;
        mult[addr_hash_idx[3]] += 16;

        mult[addr_constant_idx[0]] += 32;
        mult[addr_constant_idx[1]] += 16;
        mult[addr_constant_idx[2]] += 16;

        mult_poseidon[hash_idx[0]] += 16;
        mult_poseidon[hash_idx[1]] += 16;
        mult_poseidon[hash_idx[2]] += 16;
        mult_poseidon[hash_idx[3]] += 16;

        mult_poseidon[constant_idx[0]] += 32;
        mult_poseidon[constant_idx[1]] += 16;
        mult_poseidon[constant_idx[2]] += 16;

        let len = variables.len();
        let padded_len = len.next_power_of_two();

        for _ in len..padded_len {
            mult.push(0);
            mult_poseidon.push(0);
            a_wire.push(0);
            b_wire.push(0);
            op.push(M31::zero());
            variables.push(M31::zero());

            mult[0] += 2;
            cur_idx += 1;
        }

        let mut counts = HashMap::<usize, isize>::new();
        for (i, &v) in mult.iter().enumerate() {
            counts.insert(i, v as isize);
        }
        for &v in a_wire.iter() {
            let p = counts.get(&v).unwrap();
            counts.insert(v, *p - 1);
        }
        for &v in b_wire.iter() {
            let p = counts.get(&v).unwrap();
            counts.insert(v, *p - 1);
        }
        for (&k, &v) in counts.iter() {
            if !v.is_zero() {
                if addr_hash_idx.contains(&k) && v == 16 {
                    continue;
                } else if k == addr_constant_idx[0] && v == 32 {
                    continue;
                } else if (k == addr_constant_idx[1] || k == addr_constant_idx[2]) && v == 16 {
                    continue;
                } else {
                    assert!(
                        k == 1 && v == -1,
                        "The logUp sum is not expected: {} {}",
                        k,
                        v
                    );
                }
            }
        }

        let log_n_rows = padded_len.ilog2();
        let range = 0..(1 << log_n_rows);
        let circuit = PlonkWithAcceleratorCircuitTrace {
            mult: range.clone().map(|i| mult[i].into()).collect(),
            mult_poseidon: range.clone().map(|i| mult_poseidon[i].into()).collect(),
            a_wire: range.clone().map(|i| a_wire[i].into()).collect(),
            b_wire: range.clone().map(|i| b_wire[i].into()).collect(),
            c_wire: range.clone().map(|i| i.into()).collect(),
            op: range.clone().map(|i| op[i].into()).collect(),
            a_val: range.clone().map(|i| variables[a_wire[i]].into()).collect(),
            b_val: range.clone().map(|i| variables[b_wire[i]].into()).collect(),
            c_val: range.clone().map(|i| variables[i].into()).collect(),
        };

        let mut data_flow = PoseidonDataFlow(HashMap::new());
        data_flow.0.insert(constant_idx[0], CONSTANT_1);
        data_flow.0.insert(constant_idx[1], CONSTANT_2);
        data_flow.0.insert(constant_idx[2], CONSTANT_3);
        data_flow.0.insert(hash_idx[0], TEST_1);
        data_flow.0.insert(hash_idx[1], TEST_2);
        data_flow.0.insert(hash_idx[2], TEST_3);
        data_flow.0.insert(hash_idx[3], TEST_4);

        let mut prescribed_flow = PoseidonPrescribedFlow {
            addr_1: vec![],
            addr_2: vec![],
            addr_3: vec![],
            addr_4: vec![],
        };
        let mut control_flow = PoseidonControlFlow {
            sel_1: vec![],
            sel_2: vec![],
            sel_3: vec![],
            sel_4: vec![],
        };

        {
            for i in 0..32 {
                if i % 2 == 0 {
                    prescribed_flow.addr_1.push(addr_hash_idx[0]);
                    prescribed_flow.addr_2.push(addr_hash_idx[1]);
                    prescribed_flow.addr_3.push(addr_hash_idx[2]);
                    prescribed_flow.addr_4.push(addr_hash_idx[3]);

                    control_flow.sel_1.push(hash_idx[0]);
                    control_flow.sel_2.push(hash_idx[1]);
                    control_flow.sel_3.push(hash_idx[2]);
                    control_flow.sel_4.push(hash_idx[3]);
                } else {
                    prescribed_flow.addr_1.push(addr_constant_idx[0]);
                    prescribed_flow.addr_2.push(addr_constant_idx[0]);
                    prescribed_flow.addr_3.push(addr_constant_idx[1]);
                    prescribed_flow.addr_4.push(addr_constant_idx[2]);

                    control_flow.sel_1.push(constant_idx[0]);
                    control_flow.sel_2.push(constant_idx[0]);
                    control_flow.sel_3.push(constant_idx[1]);
                    control_flow.sel_4.push(constant_idx[2]);
                }
            }
        }

        let metadata = PoseidonMetadata {
            prescribed_flow,
            control_flow,
            data_flow,
            constant_1_addr: addr_constant_idx[0],
            constant_1_sel: constant_idx[0],
            constant_2_addr: addr_constant_idx[1],
            constant_2_sel: constant_idx[1],
            constant_3_addr: addr_constant_idx[2],
            constant_3_sel: constant_idx[2],
        };

        for r in [
            &metadata.prescribed_flow.addr_1,
            &metadata.prescribed_flow.addr_2,
            &metadata.prescribed_flow.addr_3,
            &metadata.prescribed_flow.addr_4,
        ]
        .iter()
        {
            for &v in r.iter() {
                let p = counts.get(&v).unwrap();
                counts.insert(v, *p - 1);
            }
        }
        for (&k, &v) in counts.iter() {
            if !v.is_zero() {
                assert!(
                    k == 1 && v == -1,
                    "The logUp sum is not expected: {} {}",
                    k,
                    v
                );
            }
        }

        let mut counts_poseidon = HashMap::<usize, isize>::new();
        for (i, &v) in mult_poseidon.iter().enumerate() {
            counts_poseidon.insert(i, v as isize);
        }
        for r in [
            &metadata.control_flow.sel_1,
            &metadata.control_flow.sel_2,
            &metadata.control_flow.sel_3,
            &metadata.control_flow.sel_4,
        ]
        .iter()
        {
            for &v in r.iter() {
                let p = counts_poseidon.get(&v).unwrap();
                counts_poseidon.insert(v, *p - 1);
            }
        }
        for (_, &v) in counts_poseidon.iter() {
            assert!(v.is_zero());
        }

        (circuit, metadata)
    }

    #[test]
    fn test_individual_proofs() {
        let (plonk, mut poseidon) = generate_test_circuit();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(2, 4, 64),
        };

        let (plonk_component, plonk_proof) = prove_plonk_with_accelerator::<Blake2sMerkleChannel>(
            plonk.mult.length.ilog2(),
            config,
            &plonk,
        );

        let (poseidon_component, poseidon_proof) = prove_poseidon_accelerator::<Blake2sMerkleChannel>(
            poseidon.control_flow.sel_1.len().ilog2(),
            config,
            &mut poseidon,
        );

        // If the two proofs are not generated together, their lookup parameters would be different,
        // but we can still verify them independently.

        // Verify the Plonk proof
        {
            // TODO: Create Air instance independently.
            let channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Decommit.
            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = plonk_component.trace_log_degree_bounds();

            // Preprocessed columns.
            commitment_scheme.commit(plonk_proof.commitments[0], &sizes[0], channel);

            // Trace columns.
            commitment_scheme.commit(plonk_proof.commitments[1], &sizes[1], channel);
            // Draw lookup element.
            let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);
            assert_eq!(lookup_elements, plonk_component.lookup_elements);
            // Interaction columns.
            commitment_scheme.commit(plonk_proof.commitments[2], &sizes[2], channel);

            verify(&[&plonk_component], channel, commitment_scheme, plonk_proof).unwrap();
        }

        // Verify the Poseidon proof
        {
            // TODO: Create Air instance independently.
            let channel = &mut Blake2sChannel::default();
            let commitment_scheme =
                &mut CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);

            // Decommit.
            // Retrieve the expected column sizes in each commitment interaction, from the AIR.
            let sizes = poseidon_component.trace_log_degree_bounds();

            // Preprocessed columns.
            commitment_scheme.commit(poseidon_proof.commitments[0], &sizes[0], channel);
            // Trace columns.
            commitment_scheme.commit(poseidon_proof.commitments[1], &sizes[1], channel);
            // Draw lookup element.
            let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);
            assert_eq!(lookup_elements, poseidon_component.lookup_elements);
            // Interaction columns.
            commitment_scheme.commit(poseidon_proof.commitments[2], &sizes[2], channel);

            verify(
                &[&poseidon_component],
                channel,
                commitment_scheme,
                poseidon_proof,
            )
            .unwrap();
        }
    }

    #[test]
    fn test_joint_proof() {
        let (plonk, mut poseidon) = generate_test_circuit();
        let config = PcsConfig {
            pow_bits: 10,
            fri_config: FriConfig::new(2, 4, 64),
        };

        let proof = prove_plonk_with_poseidon::<Blake2sMerkleChannel>(
            plonk.mult.length.ilog2(),
            poseidon.control_flow.sel_1.len().ilog2(),
            config,
            &plonk,
            &mut poseidon,
        );
        verify_plonk_with_poseidon::<Blake2sMerkleChannel>(proof, config).unwrap();
    }

    #[test]
    fn test_joint_proof_serialize_poseidon31() {
        let (plonk, mut poseidon) = generate_test_circuit();
        let config = PcsConfig {
            pow_bits: 20,
            fri_config: FriConfig::new(3, 5, 16),
        };

        let proof = prove_plonk_with_poseidon::<Poseidon31MerkleChannel>(
            plonk.mult.length.ilog2(),
            poseidon.control_flow.sel_1.len().ilog2(),
            config,
            &plonk,
            &mut poseidon,
        );

        let encoded = bincode::serialize(&proof).unwrap();

        let decoded: PlonkWithPoseidonProof<Poseidon31MerkleHasher> =
            bincode::deserialize(&encoded).unwrap();
        verify_plonk_with_poseidon::<Poseidon31MerkleChannel>(decoded, config).unwrap();
    }
}
