use engine0::utf8::{Utf8StreamDecoder, Utf8StreamError};

#[test]
fn partial_scalar_is_buffered_until_complete() {
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(decoder.push(&[0xe4, 0xb8]).unwrap(), None);
    assert_eq!(decoder.pending_bytes(), &[0xe4, 0xb8]);
    assert_eq!(decoder.push(&[0x96]).unwrap(), Some("世".to_string()));
    assert!(decoder.pending_bytes().is_empty());
}

#[test]
fn one_piece_can_emit_multiple_scalars() {
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(
        decoder.push("世界🚀".as_bytes()).unwrap(),
        Some("世界🚀".to_string())
    );
    assert_eq!(decoder.finish(), Ok(()));
}

#[test]
fn complete_prefix_emits_while_incomplete_suffix_stays_buffered() {
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(
        decoder.push(&[b'A', 0xf0, 0x9f]).unwrap(),
        Some("A".to_string())
    );
    assert_eq!(decoder.pending_bytes(), &[0xf0, 0x9f]);
}

#[test]
fn malformed_sequence_is_not_replaced_or_ignored() {
    let mut decoder = Utf8StreamDecoder::new();
    assert!(matches!(
        decoder.push(&[0xc3, 0x28]),
        Err(Utf8StreamError::InvalidSequence { .. })
    ));
    assert_eq!(decoder.pending_bytes(), &[0xc3, 0x28]);
}

#[test]
fn terminal_with_incomplete_suffix_is_an_error() {
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(decoder.push(&[0xf0, 0x9f]).unwrap(), None);
    assert!(matches!(
        decoder.finish(),
        Err(Utf8StreamError::IncompleteSequence { .. })
    ));
}

#[test]
fn empty_pieces_do_not_emit_empty_text() {
    let mut decoder = Utf8StreamDecoder::new();
    assert_eq!(decoder.push(b"").unwrap(), None);
    assert_eq!(decoder.finish(), Ok(()));
}
