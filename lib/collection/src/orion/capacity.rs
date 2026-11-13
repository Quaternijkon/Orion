//! Integer checks for the explicit capacity-constrained partition variant.
pub const ALGORITHM: &str = "orion-layout-v4-capacity-partition";

pub fn validate_replication(
    physical: u64,
    logical: usize,
    max_factor: Option<f64>,
) -> Result<(), &'static str> {
    if logical == 0 || (physical as u128) < logical as u128 {
        return Err("physical copies must cover every logical point");
    }
    if let Some(limit) = max_factor {
        if !limit.is_finite() || limit < 1.0 {
            return Err("maximum replication factor must be finite and at least 1");
        }
        if physical as f64 / logical as f64 > limit {
            return Err("capacity partition exceeds maximum replication factor");
        }
    }
    Ok(())
}

pub fn validate_counts(counts: &[u64]) -> Result<(), &'static str> {
    if counts.is_empty() || counts.contains(&0) {
        return Err("capacity partition must populate every shard");
    }
    let total: u128 = counts.iter().map(|&x| x as u128).sum();
    let largest = *counts.iter().max().unwrap() as u128;
    // Version4 fixes a hard upper bound of 1.2 times the physical-copy mean.
    if largest * counts.len() as u128 * 5 > total * 6 {
        return Err("capacity partition exceeds 1.2 times mean physical shard size");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn balances_physical_counts_not_only_upper_node_counts() {
        assert!(
            validate_counts(&[
                161645, 150629, 160040, 152320, 145695, 158202, 165493, 147096
            ])
            .is_ok()
        );
        assert!(
            validate_counts(&[55761, 62270, 367611, 89047, 181231, 234017, 175648, 80753]).is_err()
        );
    }
    #[test]
    fn exact_capacity_boundary_and_empty_shards() {
        assert!(validate_counts(&[120, 80]).is_ok());
        assert!(validate_counts(&[121, 79]).is_err());
        assert!(validate_counts(&[0, 1]).is_err());
        assert!(validate_counts(&[]).is_err());
    }
    #[test]
    fn large_counts_do_not_overflow_u64_arithmetic() {
        assert!(validate_counts(&[u64::MAX, u64::MAX]).is_ok());
    }
}
