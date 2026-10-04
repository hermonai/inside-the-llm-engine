//! Structural oracle for Chapter 40 proposal trees.
//! It intentionally contains no neural network. The goal is to prove that packed
//! storage order, logical position and causal ancestry are distinct coordinates.

#[derive(Clone, Debug)]
pub struct Node {
    pub token: i32,
    pub parent: Option<usize>,
    pub score: f64,
}

pub fn depth(nodes: &[Node], mut i: usize) -> usize {
    let mut d = 1;
    while let Some(p) = nodes[i].parent {
        d += 1;
        assert!(d <= nodes.len(), "parent cycle");
        i = p;
    }
    d
}
pub fn ancestors_inclusive(nodes: &[Node], mut i: usize) -> Vec<usize> {
    let mut v = vec![i];
    while let Some(p) = nodes[i].parent {
        assert!(v.len() < nodes.len(), "parent cycle");
        v.push(p);
        i = p;
    }
    v.reverse();
    v
}
pub fn ancestry_mask(nodes: &[Node]) -> Vec<Vec<bool>> {
    (0..nodes.len())
        .map(|i| {
            let mut row = vec![false; nodes.len()];
            for j in ancestors_inclusive(nodes, i) {
                row[j] = true;
            }
            row
        })
        .collect()
}
/// Wrong for trees: packed row i sees every packed row <= i.
pub fn triangular_mask(n: usize) -> Vec<Vec<bool>> {
    (0..n).map(|i| (0..n).map(|j| j <= i).collect()).collect()
}
/// Independent semantic oracle: reconstruct token history by walking parent pointers.
pub fn path_tokens(nodes: &[Node], i: usize) -> Vec<i32> {
    // Validate termination, then use a separate recursive reconstruction rather
    // than the mask builder's ancestor-index list.
    depth(nodes, i);
    fn visit(nodes: &[Node], i: usize, out: &mut Vec<i32>) {
        if let Some(parent) = nodes[i].parent {
            visit(nodes, parent, out);
        }
        out.push(nodes[i].token);
    }
    let mut out = Vec::new();
    visit(nodes, i, &mut out);
    out
}
pub fn mask_tokens(nodes: &[Node], mask: &[Vec<bool>], i: usize) -> Vec<i32> {
    mask[i]
        .iter()
        .enumerate()
        .filter_map(|(j, on)| if *on { Some(nodes[j].token) } else { None })
        .collect()
}
/// Deterministic teaching best-first expansion: returns node indices by descending
/// score among eligible nodes, preserving ancestor closure within the budget.
/// Real EAGLE-2 tree construction is richer.
pub fn best_nodes(nodes: &[Node], budget: usize) -> Vec<usize> {
    for i in 0..nodes.len() {
        depth(nodes, i);
        assert!(nodes[i].score.is_finite(), "scores must be finite");
    }
    let mut ids: Vec<_> = (0..nodes.len()).collect();
    ids.sort_by(|&a, &b| {
        nodes[b]
            .score
            .partial_cmp(&nodes[a].score)
            .unwrap()
            .then(a.cmp(&b))
    });
    let mut chosen = Vec::new();
    while chosen.len() < budget.min(nodes.len()) {
        let index = ids
            .iter()
            .position(|&i| nodes[i].parent.is_none_or(|p| chosen.contains(&p)))
            .expect("validated forest has an eligible frontier");
        chosen.push(ids.remove(index));
    }
    chosen
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_scoring_child_cannot_displace_its_ancestor() {
        let nodes = vec![
            Node {
                token: 10,
                parent: None,
                score: 0.1,
            },
            Node {
                token: 20,
                parent: Some(0),
                score: 0.99,
            },
        ];
        assert_eq!(best_nodes(&nodes, 1), vec![0]);
        assert_eq!(best_nodes(&nodes, 2), vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "parent cycle")]
    fn cyclic_tree_is_rejected_not_traversed_forever() {
        depth(
            &[Node {
                token: 10,
                parent: Some(0),
                score: 0.1,
            }],
            0,
        );
    }
    fn tree() -> Vec<Node> {
        vec![
            Node {
                token: 10,
                parent: None,
                score: 0.9,
            },
            Node {
                token: 20,
                parent: Some(0),
                score: 0.7,
            },
            Node {
                token: 30,
                parent: Some(0),
                score: 0.8,
            },
            Node {
                token: 40,
                parent: Some(1),
                score: 0.6,
            },
            Node {
                token: 50,
                parent: Some(2),
                score: 0.5,
            },
        ]
    }
    #[test]
    fn depths_follow_parents_not_rows() {
        let n = tree();
        assert_eq!(
            (0..5).map(|i| depth(&n, i)).collect::<Vec<_>>(),
            vec![1, 2, 2, 3, 3]
        );
    }
    #[test]
    fn flattened_mask_matches_independent_parent_path() {
        let n = tree();
        let m = ancestry_mask(&n);
        for i in 0..n.len() {
            assert_eq!(mask_tokens(&n, &m, i), path_tokens(&n, i));
        }
    }
    #[test]
    fn triangular_mask_leaks_sibling() {
        let n = tree();
        let m = triangular_mask(n.len());
        assert_ne!(mask_tokens(&n, &m, 2), path_tokens(&n, 2));
        assert!(mask_tokens(&n, &m, 2).contains(&20));
    }
    #[test]
    fn sibling_isolation() {
        let n = tree();
        let m = ancestry_mask(&n);
        assert!(!m[2][1]);
        assert!(!m[4][1]);
        assert!(!m[4][3]);
    }
    #[test]
    fn best_first_budget_is_deterministic() {
        let n = tree();
        assert_eq!(best_nodes(&n, 3), vec![0, 2, 1]);
    }
}
