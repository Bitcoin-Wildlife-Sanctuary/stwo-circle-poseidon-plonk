use itertools::Itertools;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use tracing::{span, Level};

use crate::constraint_framework::{
    assert_constraints, FrameworkEval, Relation, TraceLocationAllocator, INTERACTION_TRACE_IDX,
    ORIGINAL_TRACE_IDX, PREPROCESSED_TRACE_IDX,
};
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
use crate::examples::plonk_without_poseidon::plonk::{
    gen_interaction_trace, gen_trace, PlonkWithoutAcceleratorCircuitTrace,
    PlonkWithoutAcceleratorComponent, PlonkWithoutAcceleratorEval,
    PlonkWithoutAcceleratorLookupElements,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithoutPoseidonStatement0 {
    pub log_size_plonk: u32,
}

impl PlonkWithoutPoseidonStatement0 {
    pub fn log_sizes(&self) -> TreeVec<Vec<u32>> {
        let mut sizes = TreeVec::new(vec![vec![], vec![], vec![]]);
        let log_size_plonk = self.log_size_plonk;

        sizes[PREPROCESSED_TRACE_IDX].extend_from_slice(&[log_size_plonk; 7]);
        sizes[ORIGINAL_TRACE_IDX].extend_from_slice(&[log_size_plonk; 12]);
        sizes[INTERACTION_TRACE_IDX].extend_from_slice(&[log_size_plonk; 4]);

        sizes
    }

    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_u64(self.log_size_plonk as u64);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithoutPoseidonStatement1 {
    pub plonk_total_sum: SecureField,
}

impl PlonkWithoutPoseidonStatement1 {
    pub fn mix_into(&self, channel: &mut impl Channel) {
        channel.mix_felts(&[self.plonk_total_sum]);
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PlonkWithoutPoseidonProof<H: MerkleHasher> {
    pub stmt0: PlonkWithoutPoseidonStatement0,
    pub stmt1: PlonkWithoutPoseidonStatement1,
    pub stark_proof: StarkProof<H>,
}

pub fn prove_plonk_without_poseidon<MC: MerkleChannel>(
    config: PcsConfig,
    circuit: &PlonkWithoutAcceleratorCircuitTrace,
) -> PlonkWithoutPoseidonProof<MC::H>
where
    SimdBackend: BackendForChannel<MC>,
{
    let log_size_plonk = circuit.mult_c.length.ilog2();

    assert!(log_size_plonk >= LOG_N_LANES);
    assert_eq!(circuit.mult_c.length, 1 << log_size_plonk);

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let log_max_rows = log_size_plonk + 2;
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
    let plonk_constant_trace = [
        circuit.a_wire.clone(),
        circuit.b_wire.clone(),
        circuit.c_wire.clone(),
        circuit.op1.clone(),
        circuit.op2.clone(),
        circuit.op3.clone(),
        circuit.mult_c.clone(),
    ]
    .into_iter()
    .map(|eval| {
        CircleEvaluation::<SimdBackend, _, BitReversedOrder>::new(
            CanonicCoset::new(log_size_plonk).circle_domain(),
            eval.clone(),
        )
    })
    .collect_vec();

    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(plonk_constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let plonk_trace = gen_trace(log_size_plonk, &circuit);

    // Statement0.
    let stmt0 = PlonkWithoutPoseidonStatement0 { log_size_plonk };
    stmt0.mix_into(channel);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(plonk_trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithoutAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (plonk_interaction_trace, plonk_total_sum) =
        gen_interaction_trace(log_size_plonk, &circuit, &lookup_elements);

    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(plonk_interaction_trace);
    // Statement1.
    let stmt1 = PlonkWithoutPoseidonStatement1 { plonk_total_sum };
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
    let component = PlonkWithoutAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PlonkWithoutAcceleratorEval {
            log_n_rows: log_size_plonk,
            lookup_elements,
            total_sum: plonk_total_sum,
        },
        plonk_total_sum,
    );

    // Sanity check. Remove for production.
    let trace_polys = commitment_scheme
        .trees
        .as_ref()
        .map(|t| t.polynomials.iter().cloned().collect_vec());
    assert_constraints(
        &trace_polys,
        CanonicCoset::new(log_size_plonk),
        |eval| {
            component.evaluate(eval);
        },
        plonk_total_sum,
    );

    let stark_proof = prove(&[&component], channel, commitment_scheme).unwrap();

    PlonkWithoutPoseidonProof {
        stmt0,
        stmt1,
        stark_proof,
    }
}

#[allow(unused)]
pub fn verify_plonk_without_poseidon<MC: MerkleChannel>(
    PlonkWithoutPoseidonProof {
        stmt0,
        stmt1,
        stark_proof,
    }: PlonkWithoutPoseidonProof<MC::H>,
    config: PcsConfig,
    inputs: &[(usize, M31)],
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
    let lookup_elements = PlonkWithoutAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    stmt1.mix_into(channel);
    commitment_scheme.commit(stark_proof.commitments[2], &log_sizes[2], channel);

    let mut input_sum = SecureField::zero();
    for &(i, v) in inputs.iter() {
        let sum: SecureField = lookup_elements.combine(&[v, M31::from(i)]);
        input_sum += sum.inverse();
    }

    let total_sum = stmt1.plonk_total_sum + input_sum;
    assert_eq!(total_sum, SecureField::zero());

    let component = PlonkWithoutAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PlonkWithoutAcceleratorEval {
            log_n_rows: stmt0.log_size_plonk,
            lookup_elements,
            total_sum: stmt1.plonk_total_sum,
        },
        stmt1.plonk_total_sum,
    );

    verify(&[&component], channel, commitment_scheme, stark_proof)
}
