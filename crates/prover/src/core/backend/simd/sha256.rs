use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::core::backend::simd::SimdBackend;
use crate::core::backend::{Col, Column, ColumnOps};
use crate::core::fields::m31::BaseField;
use crate::core::vcs::ops::{MerkleHasher, MerkleOps};
use crate::core::vcs::sha256_hash::Sha256Hash;
use crate::core::vcs::sha256_merkle::Sha256MerkleHasher;
use crate::parallel_iter;

impl ColumnOps<Sha256Hash> for SimdBackend {
    type Column = Vec<Sha256Hash>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Sha256MerkleHasher> for SimdBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&Vec<Sha256Hash>>,
        columns: &[&Col<Self, BaseField>],
    ) -> Vec<Sha256Hash> {
        parallel_iter!(0..1 << log_size)
            .map(|i| {
                Sha256MerkleHasher::hash_node(
                    prev_layer.map(|prev_layer| (prev_layer[2 * i], prev_layer[2 * i + 1])),
                    &columns.iter().map(|column| column.at(i)).collect_vec(),
                )
            })
            .collect()
    }
}
