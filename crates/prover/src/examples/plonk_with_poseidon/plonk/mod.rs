use std::simd::Mask;

use itertools::Itertools;
use num_traits::One;
use tracing::{span, Level};

use crate::constraint_framework::logup::{LogupTraceGenerator, LookupElements};
use crate::constraint_framework::preprocessed_columns::{gen_is_first, PreprocessedColumn};
use crate::constraint_framework::{
    assert_constraints, EvalAtRow, FrameworkComponent, FrameworkEval, RelationEntry,
    TraceLocationAllocator, ORIGINAL_TRACE_IDX,
};
use crate::core::backend::simd::column::BaseColumn;
use crate::core::backend::simd::m31::{PackedM31, LOG_N_LANES};
use crate::core::backend::simd::qm31::PackedSecureField;
use crate::core::backend::simd::SimdBackend;
use crate::core::backend::Column;
use crate::core::channel::Blake2sChannel;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::{CommitmentSchemeProver, PcsConfig, TreeSubspan};
use crate::core::poly::circle::{CanonicCoset, CircleEvaluation, PolyOps};
use crate::core::poly::BitReversedOrder;
use crate::core::prover::{prove, StarkProof};
use crate::core::vcs::blake2_merkle::{Blake2sMerkleChannel, Blake2sMerkleHasher};
use crate::core::ColumnVec;
use crate::relation;

pub type PlonkWithAcceleratorComponent = FrameworkComponent<PlonkWithAcceleratorEval>;
relation!(PlonkWithAcceleratorLookupElements, 9);

#[derive(Clone)]
pub struct PlonkWithAcceleratorEval {
    pub log_n_rows: u32,
    pub lookup_elements: PlonkWithAcceleratorLookupElements,
    pub total_sum: SecureField,
    pub base_trace_location: TreeSubspan,
    pub interaction_trace_location: TreeSubspan,
    pub constants_trace_location: TreeSubspan,
}

impl FrameworkEval for PlonkWithAcceleratorEval {
    fn log_size(&self) -> u32 {
        self.log_n_rows
    }

    fn max_constraint_log_degree_bound(&self) -> u32 {
        self.log_n_rows + 1
    }

    fn evaluate<E: EvalAtRow>(&self, mut eval: E) -> E {
        let a_wire = eval.get_preprocessed_column(PreprocessedColumn::Plonk(0));
        let b_wire = eval.get_preprocessed_column(PreprocessedColumn::Plonk(1));
        // Note: c_wire could also be implicit: (self.eval.point() - M31_CIRCLE_GEN.into_ef()).x.
        //   A constant column is easier though.
        let c_wire = eval.get_preprocessed_column(PreprocessedColumn::Plonk(2));
        let op = eval.get_preprocessed_column(PreprocessedColumn::Plonk(3));
        let mult = eval.get_preprocessed_column(PreprocessedColumn::Plonk(4));
        let mult_poseidon = eval.get_preprocessed_column(PreprocessedColumn::Plonk(5));

        let a_val = eval.next_trace_mask();
        let b_val = eval.next_trace_mask();
        let c_vals = eval.next_interaction_mask(ORIGINAL_TRACE_IDX, [0, 1, 2, 3, 4, 5, 6, 7]);

        eval.add_constraint(
            c_vals[0].clone() - op.clone() * (a_val.clone() + b_val.clone())
                + (E::F::one() - op) * a_val.clone() * b_val.clone(),
        );

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[a_wire, a_val],
        ));
        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            E::EF::one(),
            &[b_wire, b_val],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            (-mult).into(),
            &[c_wire.clone(), c_vals[0].clone()],
        ));

        eval.add_to_relation(RelationEntry::new(
            &self.lookup_elements,
            mult_poseidon.into(),
            &[
                -c_wire.clone(),
                c_vals[0].clone(),
                c_vals[1].clone(),
                c_vals[2].clone(),
                c_vals[3].clone(),
                c_vals[4].clone(),
                c_vals[5].clone(),
                c_vals[6].clone(),
                c_vals[7].clone(),
            ],
        ));

        eval.finalize_logup_in_pairs();
        eval
    }
}

#[derive(Clone)]
pub struct Plonk2CircuitTrace {
    pub mult: BaseColumn,
    pub mult_poseidon: BaseColumn,
    pub a_wire: BaseColumn,
    pub b_wire: BaseColumn,
    pub c_wire: BaseColumn,
    pub op: BaseColumn,
    pub a_val: BaseColumn,
    pub b_val: BaseColumn,
    pub c_val: BaseColumn,
}
pub fn gen_trace(
    log_size: u32,
    circuit: &Plonk2CircuitTrace,
) -> ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    let _span = span!(Level::INFO, "Generation").entered();

    let coset = CanonicCoset::new(log_size);
    [&circuit.a_val, &circuit.b_val, &circuit.c_val]
        .into_iter()
        .map(|eval| CircleEvaluation::new_canonical_ordered(coset, eval.clone()))
        .collect()
}

pub fn gen_interaction_trace(
    log_size: u32,
    circuit: &Plonk2CircuitTrace,
    lookup_elements: &LookupElements<9>,
) -> (
    ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>>,
    SecureField,
) {
    let _span = span!(Level::INFO, "Generate interaction trace").entered();
    let mut logup_gen = LogupTraceGenerator::new(log_size);

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let q0: PackedSecureField =
            lookup_elements.combine(&[circuit.a_wire.data[vec_row], circuit.a_val.data[vec_row]]);
        let q1: PackedSecureField =
            lookup_elements.combine(&[circuit.b_wire.data[vec_row], circuit.b_val.data[vec_row]]);
        col_gen.write_frac(vec_row, q0 + q1, q0 * q1);
    }
    col_gen.finalize_col();

    let mut col_gen = logup_gen.new_col();
    for vec_row in 0..(1 << (log_size - LOG_N_LANES)) {
        let p0 = -circuit.mult.data[vec_row];
        let q0: PackedSecureField =
            lookup_elements.combine(&[circuit.c_wire.data[vec_row], circuit.c_val.data[vec_row]]);

        let p1 = circuit.mult_poseidon.data[vec_row];

        let mut c_val_and_shifted_ones = Vec::with_capacity(9);
        c_val_and_shifted_ones.push(-circuit.c_wire.data[vec_row]);
        c_val_and_shifted_ones.push(circuit.c_val.data[vec_row]);

        let mut first = circuit.c_val.data[vec_row].into_simd();
        let mut second =
            circuit.c_val.data[(vec_row + 1) % (1 << (log_size - LOG_N_LANES))].into_simd();

        for i in 1..8 {
            let mask = Mask::<_, 16>::from_bitmask((1 << (16 - i)) - 1);
            first = first.rotate_elements_left::<1>();
            second = second.rotate_elements_left::<1>();
            unsafe {
                c_val_and_shifted_ones.push(PackedM31::from_simd_unchecked(
                    mask.clone().select(first, second),
                ));
            }
        }
        let q1: PackedSecureField = lookup_elements.combine(&c_val_and_shifted_ones);

        col_gen.write_frac(vec_row, p0.into(), q0);
        col_gen.write_frac(vec_row, q0 * p1 + q1 * p0, q0 * q1);
    }
    col_gen.finalize_col();

    logup_gen.finalize_last_canonical()
}

#[allow(unused)]
pub fn prove_fibonacci_plonk_with_accelerator(
    log_n_rows: u32,
    config: PcsConfig,
) -> (
    PlonkWithAcceleratorComponent,
    StarkProof<Blake2sMerkleHasher>,
) {
    assert!(log_n_rows >= LOG_N_LANES);

    // Prepare a fibonacci circuit.
    let mut fib_values = vec![BaseField::one(), BaseField::one()];
    for _ in 0..(1 << log_n_rows) {
        fib_values.push(fib_values[fib_values.len() - 1] + fib_values[fib_values.len() - 2]);
    }
    let range = 0..(1 << log_n_rows);
    let mut circuit = Plonk2CircuitTrace {
        mult: range.clone().map(|_| 2.into()).collect(),
        mult_poseidon: range.clone().map(|_| 0.into()).collect(),
        a_wire: range.clone().map(|i| i.into()).collect(),
        b_wire: range.clone().map(|i| (i + 1).into()).collect(),
        c_wire: range.clone().map(|i| (i + 2).into()).collect(),
        op: range.clone().map(|_| 1.into()).collect(),
        a_val: range.clone().map(|i| fib_values[i]).collect(),
        b_val: range.clone().map(|i| fib_values[i + 1]).collect(),
        c_val: range.clone().map(|i| fib_values[i + 2]).collect(),
    };
    circuit.mult_poseidon.set(1, 1.into());
    circuit.mult.set((1 << log_n_rows) - 1, 0.into());
    circuit.mult.set((1 << log_n_rows) - 2, 1.into());

    // Precompute twiddles.
    let span = span!(Level::INFO, "Precompute twiddles").entered();
    let twiddles = SimdBackend::precompute_twiddles(
        CanonicCoset::new(log_n_rows + config.fri_config.log_blowup_factor + 1)
            .circle_domain()
            .half_coset,
    );
    span.exit();

    // Setup protocol.
    let channel = &mut Blake2sChannel::default();
    let mut commitment_scheme =
        CommitmentSchemeProver::<_, Blake2sMerkleChannel>::new(config, &twiddles);

    // Preprocessed trace.
    let span = span!(Level::INFO, "Constant").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    let is_first = gen_is_first(log_n_rows);
    let mut constant_trace = [
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
            CanonicCoset::new(log_n_rows),
            col,
        )
    })
    .collect_vec();
    constant_trace.push(is_first);
    let constants_trace_location = tree_builder.extend_evals(constant_trace);
    tree_builder.commit(channel);
    span.exit();

    // Trace.
    let span = span!(Level::INFO, "Trace").entered();
    let trace = gen_trace(log_n_rows, &circuit);
    let mut tree_builder = commitment_scheme.tree_builder();
    let base_trace_location = tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();

    // Draw lookup element.
    let lookup_elements = PlonkWithAcceleratorLookupElements::draw(channel);

    // Interaction trace.
    let span = span!(Level::INFO, "Interaction").entered();
    let (trace, total_sum) = gen_interaction_trace(log_n_rows, &circuit, &lookup_elements.0);
    let mut tree_builder = commitment_scheme.tree_builder();
    let interaction_trace_location = tree_builder.extend_evals(trace);
    tree_builder.commit(channel);
    span.exit();
    // Prove constraints.
    let component = PlonkWithAcceleratorComponent::new(
        &mut TraceLocationAllocator::default(),
        PlonkWithAcceleratorEval {
            log_n_rows,
            lookup_elements,
            total_sum,
            base_trace_location,
            interaction_trace_location,
            constants_trace_location,
        },
        (total_sum, None),
    );

    // Sanity check. Remove for production.
    let trace_polys = commitment_scheme
        .trees
        .as_ref()
        .map(|t| t.polynomials.iter().cloned().collect_vec());
    assert_constraints(
        &trace_polys,
        CanonicCoset::new(log_n_rows),
        |mut eval| {
            component.evaluate(eval);
        },
        (total_sum, None),
    );

    let proof = prove(&[&component], channel, commitment_scheme).unwrap();

    (component, proof)
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
    use crate::examples::plonk_with_poseidon::plonk::{
        prove_fibonacci_plonk_with_accelerator, PlonkWithAcceleratorLookupElements,
    };

    #[test_log::test]
    fn test_simd_plonk_with_accelerator_prove() {
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
        let (component, proof) = prove_fibonacci_plonk_with_accelerator(log_n_instances, config);

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
