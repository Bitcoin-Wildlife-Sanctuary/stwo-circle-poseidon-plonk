use num_traits::Zero;
use sha2::{Digest, Sha256};

use crate::core::channel::Channel;
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::vcs::bitcoin_num_to_bytes;
use crate::core::vcs::sha256_hash::{Sha256Hash, Sha256Hasher};

pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a SHA256 hash.
#[derive(Clone, Debug, Default)]
pub struct Sha256Channel {
    digest: Sha256Hash,
    n_draws: u32,
}

impl Sha256Channel {
    pub const fn digest(&self) -> Sha256Hash {
        self.digest
    }
    pub fn update_digest(&mut self, new_digest: Sha256Hash) {
        self.digest = new_digest;
        self.n_draws = 0;
    }

    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        let mut extract = [0u8; 32];

        let mut hasher = Sha256::new();
        Digest::update(&mut hasher, self.digest);
        Digest::update(&mut hasher, self.n_draws.to_le_bytes());
        extract.copy_from_slice(hasher.finalize().as_slice());

        let mut res = [BaseField::zero(); FELTS_PER_HASH];
        for i in 0..FELTS_PER_HASH {
            res[i] = extract_common(&extract[i * 4..]);
        }
        self.n_draws += 1;
        res
    }
}

impl Channel for Sha256Channel {
    const BYTES_PER_HASH: usize = 32;

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        let mut hasher = Sha256::new();
        Digest::update(&mut hasher, self.digest);
        Digest::update(&mut hasher, &nonce.to_le_bytes());
        let res = hasher.finalize();
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&res[0..16]);
        let n_zeros = u128::from_be_bytes(bytes).trailing_zeros();
        n_zeros >= n_bits
    }

    fn mix_felts(&mut self, felts: &[SecureField]) {
        for felt in felts.iter() {
            let mut hasher = Sha256::new();
            Digest::update(&mut hasher, sha256_qm31(felt));
            Digest::update(&mut hasher, self.digest);
            self.update_digest(hasher.finalize().as_slice().into());
        }
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_u32s(&[value as u32, (value >> 32) as u32]);
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        for chunk in data.chunks(8) {
            let mut hash = [0u8; 32];
            for i in 0..chunk.len() {
                hash[i * 4..(i + 1) * 4].copy_from_slice(&chunk[i].to_le_bytes());
            }
            self.digest = Sha256Hasher::concat_and_hash(&Sha256Hash(hash), &self.digest);
        }
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let res = self.draw_base_felts();
        SecureField::from_m31(res[0], res[1], res[2], res[3])
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let mut res = Vec::with_capacity(n_felts + 1);
        for _ in 0..n_felts.div_ceil(2) {
            let t = self.draw_base_felts();
            res.push(SecureField::from_m31(t[0], t[1], t[2], t[3]));
            res.push(SecureField::from_m31(t[4], t[5], t[6], t[7]));
        }
        res.truncate(n_felts);
        res
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        let mut hash_input = self.digest.0.to_vec();

        // Append counter bytes directly (4 bytes for u32).
        let counter_bytes = self.n_draws.to_le_bytes();
        hash_input.extend_from_slice(&counter_bytes);

        // Append a zero byte for domain separation between generating randomness and mixing a
        // single u32.
        hash_input.push(0_u8);

        let mut hasher = Sha256::new();
        Digest::update(&mut hasher, &hash_input);
        let extract = hasher.finalize();
        self.n_draws += 1;

        extract.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect()
    }
}

pub(crate) fn extract_common(hash: &[u8]) -> M31 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&hash[0..4]);

    let mut res = u32::from_le_bytes(bytes);
    res &= 0x7fffffff;
    res %= (1 << 31) - 1;

    M31::from(res)
}

pub fn sha256_qm31(v: &QM31) -> [u8; 32] {
    let mut res = [0u8; 32];

    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, bitcoin_num_to_bytes(v.0 .0));
    res.copy_from_slice(hasher.finalize().as_slice());

    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, bitcoin_num_to_bytes(v.0 .1));
    Digest::update(&mut hasher, res);
    res.copy_from_slice(hasher.finalize().as_slice());

    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, bitcoin_num_to_bytes(v.1 .0));
    Digest::update(&mut hasher, res);
    res.copy_from_slice(hasher.finalize().as_slice());

    let mut hasher = Sha256::new();
    Digest::update(&mut hasher, bitcoin_num_to_bytes(v.1 .1));
    Digest::update(&mut hasher, res);
    res.copy_from_slice(hasher.finalize().as_slice());

    res
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::core::channel::sha256::Sha256Channel;
    use crate::core::channel::Channel;
    use crate::core::fields::qm31::SecureField;
    use crate::m31;

    #[test]
    fn test_n_draws() {
        let mut channel = Sha256Channel::default();

        assert_eq!(channel.n_draws, 0);

        channel.draw_u32s();
        assert_eq!(channel.n_draws, 1);

        channel.draw_secure_felts(9);
        assert_eq!(channel.n_draws, 6);
    }

    #[test]
    fn test_draw_u32s() {
        let mut channel = Sha256Channel::default();

        let first_random_u32s = channel.draw_u32s();

        // Assert that next random u32s are different.
        assert_ne!(first_random_u32s, channel.draw_u32s());
    }

    #[test]
    pub fn test_draw_felt() {
        let mut channel = Sha256Channel::default();

        let first_random_felt = channel.draw_secure_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_secure_felt());
    }

    #[test]
    pub fn test_draw_felts() {
        let mut channel = Sha256Channel::default();

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
        let mut channel = Sha256Channel::default();
        let initial_digest = channel.digest();
        let felts: Vec<SecureField> = (0..2)
            .map(|i| SecureField::from(m31!(i + 1923782)))
            .collect();

        channel.mix_felts(felts.as_slice());

        assert_ne!(initial_digest, channel.digest());
    }
}
