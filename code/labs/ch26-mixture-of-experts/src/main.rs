use ch26_mixture_of_experts::{expected_union, grouped, occupancy, Assignment};

fn main() {
    for b in [1usize, 8, 32, 64, 128] {
        println!(
            "B={b:3}: expected union {:0.2}/128",
            expected_union(128, 8, b)
        );
    }

    let tokens = vec![1.0, 2.0, -1.0];
    let assignments = vec![
        Assignment {
            token: 0,
            expert: 2,
            gate: 0.7,
        },
        Assignment {
            token: 0,
            expert: 0,
            gate: 0.3,
        },
        Assignment {
            token: 1,
            expert: 1,
            gate: 0.6,
        },
        Assignment {
            token: 1,
            expert: 2,
            gate: 0.4,
        },
        Assignment {
            token: 2,
            expert: 2,
            gate: 0.8,
        },
        Assignment {
            token: 2,
            expert: 3,
            gate: 0.2,
        },
    ];
    println!("occupancy = {:?}", occupancy(4, &assignments));
    println!("combined outputs = {:?}", grouped(&tokens, &assignments));
}
