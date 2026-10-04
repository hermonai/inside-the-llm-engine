use ch20_structured_output::{vocab, CachedMask, Parser};
fn main() {
    let mut p = Parser::new();
    let mut c = CachedMask::new();
    for piece in [br#"{"ok":"#.as_slice(), b"tr".as_slice(), b"ue}".as_slice()] {
        let m = c.mask(&p);
        println!(
            "state {:?}: allowed {:?}",
            p.state(),
            m.iter()
                .map(|i| String::from_utf8_lossy(vocab()[*i]))
                .collect::<Vec<_>>()
        );
        assert!(p.feed(piece));
    }
    println!(
        "accepting={} output={}",
        p.accepting(),
        String::from_utf8_lossy(p.committed())
    );
}
