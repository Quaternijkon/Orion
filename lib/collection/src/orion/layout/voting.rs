use super::Result;
use std::collections::{BTreeMap, HashSet};

/// Ordered hits refer to positions in the immutable upper tier, not point copies.
pub fn strict_targets(hits: &[usize], owners: &[u32]) -> Result<Vec<u32>> {
    let (mut targets, max_vote) = highest_vote_targets(hits, owners)?;
    if max_vote == 1 {
        targets.truncate(1);
    }
    Ok(targets)
}

/// All highest-vote owners in nearest-hit order, including all-one ties.
fn highest_vote_targets(hits: &[usize], owners: &[u32]) -> Result<(Vec<u32>, usize)> {
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
    let mut result = Vec::new();
    for &hit in hits {
        let owner = owners[hit];
        if votes[&owner] == max_vote && !result.contains(&owner) {
            result.push(owner);
        }
    }
    Ok((result, max_vote))
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
            .enumerate()
            .map(|(i, row)| {
                let (targets, _) = highest_vote_targets(row, &current)?;
                Ok(if targets.contains(&current[i]) {
                    current[i]
                } else {
                    targets[0]
                })
            })
            .collect::<Result<_>>()?;
        if next == current {
            return Ok((current, iteration));
        }
        current = next;
    }
    // The iteration budget bounds refinement, not whether the layout is usable.
    // Preserve the last synchronous update even when labels are still changing.
    Ok((current, max_iterations))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_vote_keeps_current_owner_in_highest_vote_tie() {
        let owners = [0, 1, 0, 1];
        let hits = vec![vec![1, 0, 3, 2]; 4];
        assert_eq!(
            refine_owners(&hits, &owners, 5).unwrap(),
            (owners.to_vec(), 1)
        );
    }

    #[test]
    fn self_vote_keeps_current_owner_in_all_one_tie() {
        let owners = [0, 1, 2];
        let hits = vec![vec![2, 1, 0]; 3];
        assert_eq!(
            refine_owners(&hits, &owners, 5).unwrap(),
            (owners.to_vec(), 1)
        );
        // Ordinary data placement still chooses only the nearest owner.
        assert_eq!(strict_targets(&hits[0], &owners).unwrap(), vec![2]);
    }

    #[test]
    fn self_vote_moves_to_first_winner_when_current_owner_is_not_a_winner() {
        let owners = [2, 0, 1, 0, 1];
        let hits = vec![vec![2, 1, 4, 3, 0], vec![1], vec![2], vec![3], vec![4]];
        assert_eq!(
            refine_owners(&hits, &owners, 1).unwrap(),
            (vec![1, 0, 1, 0, 1], 1)
        );
    }

    #[test]
    fn self_vote_returns_last_synchronous_round_on_nonconvergence() {
        let hits = [vec![1], vec![0]];
        assert_eq!(refine_owners(&hits, &[0, 1], 3).unwrap(), (vec![1, 0], 3));
        assert_eq!(refine_owners(&hits, &[0, 1], 4).unwrap(), (vec![0, 1], 4));
    }

    #[test]
    fn data_placement_preserves_all_maximum_winners_and_nearest_order() {
        let owners = [0, 1, 0, 1, 2];
        assert_eq!(
            strict_targets(&[3, 0, 2, 1, 4], &owners).unwrap(),
            vec![1, 0]
        );
        assert_eq!(strict_targets(&[4, 0, 1], &owners).unwrap(), vec![2]);
    }

    #[test]
    fn invalid_inputs_still_fail_and_last_round_fixed_point_succeeds() {
        assert!(refine_owners(&[vec![]], &[0], 3).is_err());
        assert!(refine_owners(&[vec![1]], &[0], 3).is_err());
        assert!(refine_owners(&[vec![0, 0]], &[0], 3).is_err());
        assert!(refine_owners(&[vec![0]], &[0], 0).is_err());
        assert!(refine_owners(&[], &[0], 3).is_err());
        assert_eq!(
            refine_owners(&[vec![1], vec![1]], &[0, 1], 1).unwrap(),
            (vec![1, 1], 1)
        );
    }
}
