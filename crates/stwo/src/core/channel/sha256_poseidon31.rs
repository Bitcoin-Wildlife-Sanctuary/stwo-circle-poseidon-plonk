use itertools::Itertools;
use num_traits::Zero;

use crate::core::channel::{Channel, Sha256Channel};
use crate::core::fields::m31::M31;
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::vcs::poseidon31_hash::Poseidon31Hash;
use crate::core::vcs::poseidon31_merkle::Poseidon31MerkleHasher;
use crate::core::vcs::poseidon31_ref::Poseidon31CRH;
use crate::core::vcs::sha256_hash::Sha256Hash;

#[derive(Clone, Debug, Default)]
pub struct Sha256Poseidon31Channel {
    pub(crate) inner: Sha256Channel,
}

impl Sha256Poseidon31Channel {
    pub const fn digest(&self) -> Sha256Hash {
        self.inner.digest()
    }

    pub fn update_digest(&mut self, new_digest: Sha256Hash) {
        self.inner.update_digest(new_digest);
    }
}

impl Channel for Sha256Poseidon31Channel {
    const BYTES_PER_HASH: usize = 32;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        self.inner.verify_pow_nonce(n_bits, nonce)
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        if felts.len() <= 2 {
            self.inner.mix_felts(felts);
        } else {
            let elems = felts.iter().flat_map(|v| v.to_m31_array()).collect_vec();
            let poseidon_hash = Poseidon31MerkleHasher::hash_column_get_capacity(&elems);

            let mut state = [M31::zero(); 16];
            for i in 0..8 {
                state[i + 8] = poseidon_hash.0[i];
            }
            let poseidon_hash = Poseidon31Hash(Poseidon31CRH::permute_get_rate(&state));

            self.inner.mix_felts(&[
                QM31::from_m31_array(poseidon_hash.0[0..4].try_into().unwrap()),
                QM31::from_m31_array(poseidon_hash.0[4..8].try_into().unwrap()),
            ]);
        }
    }

    fn mix_u64(&mut self, value: u64) {
        self.inner.mix_u64(value);
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        self.inner.mix_u32s(data);
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        self.inner.draw_secure_felt()
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        self.inner.draw_secure_felts(n_felts)
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        self.inner.draw_u32s()
    }
}
