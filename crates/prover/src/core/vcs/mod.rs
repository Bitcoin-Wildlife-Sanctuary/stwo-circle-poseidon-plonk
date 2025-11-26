//! Vector commitment scheme (VCS) module.

use crate::core::fields::m31::M31;

pub mod blake2_hash;
pub mod blake2_merkle;
pub mod blake3_hash;
pub mod hash;
mod merkle_hasher;
pub use merkle_hasher::MerkleHasher;
#[cfg(not(target_arch = "wasm32"))]
pub mod poseidon252_merkle;
pub mod poseidon31_hash;
pub mod poseidon31_merkle;
pub mod poseidon31_ref;
pub mod sha256_hash;
pub mod sha256_merkle;
pub mod sha256_poseidon31_merkle;
pub mod utils;
pub mod verifier;

#[cfg(all(test, feature = "prover"))]
pub mod test_utils;

pub fn bitcoin_num_to_bytes(v: M31) -> Vec<u8> {
    let mut bytes = Vec::new();

    let mut v = v.0;
    while v > 0 {
        bytes.push((v & 0xff) as u8);
        v >>= 8;
    }

    if bytes.last().is_some() && bytes.last().unwrap() & 0x80 != 0 {
        bytes.push(0);
    }

    bytes
}
