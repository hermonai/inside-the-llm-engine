use ch19_scheduling::{Req, Sim};
fn main() {
    let mut s = Sim::new(6, 2);
    s.submit(
        Req {
            id: 1,
            tenant: 0,
            blocks: 3,
            remaining: 6,
            priority: 0,
            age: 0,
        },
        8,
    );
    s.submit(
        Req {
            id: 2,
            tenant: 1,
            blocks: 3,
            remaining: 4,
            priority: 0,
            age: 0,
        },
        8,
    );
    for k in 0..8 {
        s.tick();
        println!(
            "tick {k}: used={} service={:?} completed={:?}",
            s.used, s.service, s.completed
        );
    }
}
