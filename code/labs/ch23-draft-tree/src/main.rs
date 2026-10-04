use ch23_draft_tree::*;
fn main() {
    let n = vec![
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
    ];
    let m = ancestry_mask(&n);
    for i in 0..n.len() {
        println!(
            "row {i}: depth={} path={:?} visible={:?}",
            depth(&n, i),
            path_tokens(&n, i),
            mask_tokens(&n, &m, i)
        );
    }
    println!("budget 3: {:?}", best_nodes(&n, 3));
}
