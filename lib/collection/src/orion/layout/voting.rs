use super::Result;
use std::collections::{BTreeMap, HashSet};

/// Ordered hits refer to positions in the immutable upper tier, not point copies.
pub fn strict_targets(hits: &[usize], owners: &[u32]) -> Result<Vec<u32>> {
    let mut votes = BTreeMap::new();
    let mut seen = HashSet::new();
    for &hit in hits {
        let owner = *owners.get(hit).ok_or("unknown upper hit")?;
        if !seen.insert(hit) {
            return Err("duplicate upper hit".into());
        }
        *votes.entry(owner).or_insert(0_usize) += 1;
    }
    let max_vote = *votes.values().max().ok_or("empty navigation result")?;
    if max_vote == 1 {
        return Ok(vec![owners[hits[0]]]);
    }
    let mut result = Vec::new();
    for &hit in hits {
        let owner = owners[hit];
        if votes[&owner] == max_vote && !result.contains(&owner) {
            result.push(owner);
        }
    }
    Ok(result)
}

pub fn refine_owners(
    hits: &[Vec<usize>],
    owners: &[u32],
    max_iterations: usize,
) -> Result<(Vec<u32>, usize)> {
    if owners.is_empty() || hits.len() != owners.len() || max_iterations == 0 {
        return Err("invalid self-vote input".into());
    }
    let mut current = owners.to_vec();
    for iteration in 1..=max_iterations {
        let next: Vec<u32> = hits
            .iter()
            .map(|row| Ok(strict_targets(row, &current)?[0]))
            .collect::<Result<_>>()?;
        if next == current {
            return Ok((current, iteration));
        }
        current = next;
    }
    // Match the old boundary behavior: a fixed point reached on the last update succeeds.
    if hits
        .iter()
        .enumerate()
        .all(|(i, row)| strict_targets(row, &current).is_ok_and(|s| s[0] == current[i]))
    {
        return Ok((current, max_iterations));
    }
    Err(
        format!("upper self-vote did not converge within {max_iterations} synchronous rounds")
            .into(),
    )
}
