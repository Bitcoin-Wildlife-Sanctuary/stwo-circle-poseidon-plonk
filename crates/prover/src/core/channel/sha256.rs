use num_traits::Zero;
use sha2::{Digest, Sha256};

use crate::core::channel::{Channel, ChannelTime};
use crate::core::fields::m31::{BaseField, M31};
use crate::core::fields::qm31::{SecureField, QM31};
use crate::core::vcs::bitcoin_num_to_bytes;
use crate::core::vcs::sha256_hash::{Sha256Hash, Sha256Hasher};

pub const FELTS_PER_HASH: usize = 8;

/// A channel that can be used to draw random elements from a SHA256 hash.
#[derive(Clone, Default)]
pub struct Sha256Channel {
    digest: Sha256Hash,
    pub channel_time: ChannelTime,
}

impl Sha256Channel {
    pub const fn digest(&self) -> Sha256Hash {
        self.digest
    }
    pub fn update_digest(&mut self, new_digest: Sha256Hash) {
        self.digest = new_digest;
        self.channel_time.inc_challenges();
    }

    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        let mut extract = [0u8; 32];

        let mut hasher = Sha256::new();
        Digest::update(&mut hasher, self.digest);
        Digest::update(&mut hasher, self.channel_time.n_sent.to_le_bytes());
        extract.copy_from_slice(hasher.finalize().as_slice());

        let mut res = [BaseField::zero(); FELTS_PER_HASH];
        for i in 0..FELTS_PER_HASH {
            res[i] = extract_common(&extract[i * 4..]);
        }
        self.channel_time.inc_sent();
        res
    }
}

impl Channel for Sha256Channel {
    const BYTES_PER_HASH: usize = 32;

    fn trailing_zeros(&self) -> u32 {
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&self.digest.0[0..16]);
        u128::from_le_bytes(bytes).trailing_zeros()
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
        let mut hash = [0u8; 32];
        hash[..8].copy_from_slice(&value.to_le_bytes());
        self.digest = Sha256Hasher::concat_and_hash(&Sha256Hash(hash), &self.digest);
    }

    fn draw_felt(&mut self) -> SecureField {
        let res = self.draw_base_felts();
        SecureField::from_m31(res[0], res[1], res[2], res[3])
    }

    fn draw_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let mut res = Vec::with_capacity(n_felts + 1);
        for _ in 0..n_felts.div_ceil(2) {
            let t = self.draw_base_felts();
            res.push(SecureField::from_m31(t[0], t[1], t[2], t[3]));
            res.push(SecureField::from_m31(t[4], t[5], t[6], t[7]));
        }
        res.truncate(n_felts);
        res
    }

    fn draw_random_bytes(&mut self) -> Vec<u8> {
        let mut extract = [0u8; 32];

        let mut hasher = Sha256::new();
        Digest::update(&mut hasher, self.digest);
        Digest::update(&mut hasher, self.channel_time.n_sent.to_le_bytes());
        extract.copy_from_slice(hasher.finalize().as_slice());
        self.channel_time.inc_sent();

        extract.to_vec()
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
    fn test_channel_time() {
        let mut channel = Sha256Channel::default();

        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 0);

        channel.draw_random_bytes();
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 1);

        channel.draw_felts(9);
        assert_eq!(channel.channel_time.n_challenges, 0);
        assert_eq!(channel.channel_time.n_sent, 6);
    }

    #[test]
    fn test_draw_random_bytes() {
        let mut channel = Sha256Channel::default();

        let first_random_bytes = channel.draw_random_bytes();

        // Assert that next random bytes are different.
        assert_ne!(first_random_bytes, channel.draw_random_bytes());
    }

    #[test]
    pub fn test_draw_felt() {
        let mut channel = Sha256Channel::default();

        let first_random_felt = channel.draw_felt();

        // Assert that next random felt is different.
        assert_ne!(first_random_felt, channel.draw_felt());
    }

    #[test]
    pub fn test_draw_felts() {
        let mut channel = Sha256Channel::default();

        let mut random_felts = channel.draw_felts(5);
        random_felts.extend(channel.draw_felts(4));

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
