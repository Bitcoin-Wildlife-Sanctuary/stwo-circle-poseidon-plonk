use std::iter;

use num_traits::Zero;

use crate::core::channel::Channel;
use crate::core::fields::m31::{BaseField, M31, P};
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::vcs::poseidon31_ref::poseidon2_permute;

pub const POSEIDON31_BYTES_PER_HASH: usize = 32;
pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a Poseidon31 hash.
///
/// Note: This channel does not support domain separation like Blake2s and SHA256 channels.
/// The Poseidon31 hash function does not have a built-in mechanism for domain separation,
/// so all operations use the same hash context.
#[derive(Debug, Clone, Default)]
pub struct Poseidon31Channel {
    digest: [M31; 8],
    n_draws: u32,
}

impl Poseidon31Channel {
    pub const fn digest(&self) -> [M31; 8] {
        self.digest
    }
    pub fn update_digest(&mut self, new_digest: [M31; 8]) {
        self.digest = new_digest;
        self.n_draws = 0;
    }

    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        assert!((self.n_draws as u64) < (P as u64));
        let zero = M31::zero();
        let n_sent = M31::from(self.n_draws);
        let mut state = [
            n_sent,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];

        poseidon2_permute(&mut state);
        self.n_draws += 1;

        // extract elements from the first 8 elements, not the last 8 elements
        state.first_chunk::<8>().unwrap().clone()
    }
}

impl Channel for Poseidon31Channel {
    const BYTES_PER_HASH: usize = POSEIDON31_BYTES_PER_HASH;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        // For Poseidon31, we compute H(digest, nonce) and check trailing zeros
        let zero = M31::zero();
        let mut state = [
            M31::from((nonce & ((1 << 22) - 1)) as u32),
            M31::from(((nonce >> 22) & ((1 << 21) - 1)) as u32),
            M31::from(((nonce >> 43) & ((1 << 21) - 1)) as u32),
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];
        poseidon2_permute(&mut state);
        let result = state.last_chunk::<8>().unwrap();
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&result[0].0.to_le_bytes());
        bytes[4..8].copy_from_slice(&result[1].0.to_le_bytes());
        bytes[8..12].copy_from_slice(&result[2].0.to_le_bytes());
        bytes[12..16].copy_from_slice(&result[3].0.to_le_bytes());
        let n_zeros = u128::from_le_bytes(bytes).trailing_zeros();
        n_zeros >= n_bits
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        let zero = M31::zero();
        let mut state = [
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];

        for chunk in felts.chunks(2) {
            let felts = chunk[0].to_m31_array();
            for i in 0..4 {
                state[i] = felts[i];
            }

            if chunk.len() == 2 {
                let felts = chunk[1].to_m31_array();
                for i in 0..4 {
                    state[i + 4] = felts[i];
                }
            } else {
                for i in 0..4 {
                    state[i + 4] = zero;
                }
            }

            poseidon2_permute(&mut state);
        }

        let new_digest = state.last_chunk::<8>().unwrap();
        self.update_digest(*new_digest);
    }

    fn mix_u64(&mut self, value: u64) {
        let zero = M31::zero();
        let n1 = value % ((1 << 22) - 1); // 22 bits
        let n2 = (value >> 22) & ((1 << 21) - 1); // 21 bits
        let n3 = (value >> 43) & ((1 << 21) - 1); // 21 bits

        let mut state = [
            M31::from(n1 as u32),
            M31::from(n2 as u32),
            M31::from(n3 as u32),
            zero,
            zero,
            zero,
            zero,
            zero,
            self.digest[0],
            self.digest[1],
            self.digest[2],
            self.digest[3],
            self.digest[4],
            self.digest[5],
            self.digest[6],
            self.digest[7],
        ];
        poseidon2_permute(&mut state);

        let new_digest = state.last_chunk::<8>().unwrap();
        self.update_digest(*new_digest);
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        for i in data {
            self.mix_u64(*i as u64);
        }
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let felts: [BaseField; FELTS_PER_HASH] = self.draw_base_felts();
        SecureField::from_m31_array(felts[..SECURE_EXTENSION_DEGREE].try_into().unwrap())
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let mut felts = iter::from_fn(|| Some(self.draw_base_felts())).flatten();
        let secure_felts = iter::from_fn(|| {
            Some(SecureField::from_m31_array([
                felts.next()?,
                felts.next()?,
                felts.next()?,
                felts.next()?,
            ]))
        });
        secure_felts.take(n_felts).collect()
    }

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        // the implementation here is based on the assumption that the only place draw_random_bytes
        // will be used is in generating the queries, where only the lowest n bits of every 4 bytes
        // slice would be used.

        let felts: [BaseField; FELTS_PER_HASH] = self.draw_base_felts();
        let mut bytes = Vec::with_capacity(FELTS_PER_HASH * 4);
        for i in 0..FELTS_PER_HASH {
            // important: only le bytes
            bytes.extend(felts[i].0.to_le_bytes());
        }

        bytes
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::core::channel::{Channel, Poseidon31Channel};
    use crate::core::fields::qm31::SecureField;
    use crate::m31;

    #[test]
    fn test_n_draws() {
        let mut channel = Poseidon31Channel::default();

        assert_eq!(channel.n_draws, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.n_draws, 1);

        channel.draw_secure_felts(9);
        assert_eq!(channel.n_draws, 6);
    }

    #[test]
    fn test_draw_random_bytes() {
        let mut channel = Poseidon31Channel::default();

        let first_random_bytes = channel.draw_random_bytes();

        // Assert that next random bytes are different.
        assert_ne!(first_random_bytes, channel.draw_random_bytes());
    }

    #[test]
    pub fn test_draw_felt() {
        let mut channel = Poseidon31Channel::default();

        let first_random_felt = channel.draw_secure_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_secure_felt());
    }

    #[test]
    pub fn test_draw_felts() {
        let mut channel = Poseidon31Channel::default();

        let mut random_felts = channel.draw_secure_felts(5);
        random_felts.extend(channel.draw_secure_felts(4));

        // Assert that all the random felts are unique.
        assert_eq!(
            random_felts.len(),
            random_felts.iter().collect::<BTreeSet<_>>().len()
        );
    }

    #[test]
    pub fn test_mix_felts() {
        let mut channel = Poseidon31Channel::default();
        let initial_digest = channel.digest();
        let felts: Vec<SecureField> = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest());
    }
}
