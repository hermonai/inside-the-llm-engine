use engine0::tokenizer::{
    ByteTokenizer, MergeRule, SpecialToken, TinyBpeTokenizer, TinyLmTokenizer, TokenId, Tokenizer,
    TokenizerError, TINY_LM_EOS, TINY_LM_I, TINY_LM_LIKE, TINY_LM_RUST, TOKEN_ASSISTANT, TOKEN_LO,
    TOKEN_LOWER,
};

fn round_trip(tokenizer: &impl Tokenizer, input: &[u8]) {
    let ids = tokenizer.encode(input).expect("encode");
    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, input);
}

#[test]
fn byte_oracle_covers_empty_text_and_arbitrary_bytes() {
    round_trip(&ByteTokenizer, b"");
    round_trip(&ByteTokenizer, &[0, 1, 0x7f, 0x80, 0xff]);
}

#[test]
fn bpe_round_trips_required_text_fixtures() {
    let tokenizer = TinyBpeTokenizer::teaching();
    let fixtures: &[&[u8]] = &[
        b"",
        b"plain ASCII",
        "模型把文本变成编号。".as_bytes(),
        "👩🏽‍💻🚀".as_bytes(),
        "e\u{301} and é".as_bytes(),
        b"  leading\tand  internal\ntrailing  ",
        b"repeat repeat repeat",
        &[0xf0, 0x28, 0x8c, 0x28],
    ];
    for fixture in fixtures {
        round_trip(&tokenizer, fixture);
    }
}

#[test]
fn lower_follows_the_hand_merge_order() {
    let tokenizer = TinyBpeTokenizer::teaching();
    assert_eq!(tokenizer.encode(b"lower").unwrap(), vec![TOKEN_LOWER]);
    assert_eq!(tokenizer.decode_token(TOKEN_LOWER).unwrap(), b"lower");
}

#[test]
fn equal_rank_occurrences_merge_leftmost_first_deterministically() {
    let tokenizer = TinyBpeTokenizer::teaching();
    assert_eq!(tokenizer.encode(b"lolo").unwrap(), vec![TOKEN_LO, TOKEN_LO]);
    assert_eq!(
        tokenizer.encode(b"lolo").unwrap(),
        tokenizer.encode(b"lolo").unwrap()
    );
}

#[test]
fn unmergeable_input_remains_byte_ids() {
    let tokenizer = TinyBpeTokenizer::teaching();
    assert_eq!(
        tokenizer.encode(b"xyz").unwrap(),
        vec![
            TokenId(b'x' as u32),
            TokenId(b'y' as u32),
            TokenId(b'z' as u32)
        ]
    );
}

#[test]
fn marker_like_user_text_is_not_a_special_token() {
    let tokenizer = TinyBpeTokenizer::teaching();
    let ids = tokenizer.encode(b"<|assistant|>").unwrap();
    assert!(!ids.contains(&TOKEN_ASSISTANT));
    assert_eq!(tokenizer.decode(&ids).unwrap(), b"<|assistant|>");
}

#[test]
fn explicit_special_is_disjoint_from_ordinary_decode() {
    let tokenizer = TinyBpeTokenizer::teaching();
    assert_eq!(
        tokenizer.special_id(SpecialToken::Assistant),
        Some(TOKEN_ASSISTANT)
    );
    assert_eq!(
        tokenizer.decode_token(TOKEN_ASSISTANT),
        Err(TokenizerError::SpecialTokenHasNoOrdinaryBytes(
            TOKEN_ASSISTANT
        ))
    );
}

#[test]
fn invalid_token_id_is_rejected() {
    let tokenizer = TinyBpeTokenizer::teaching();
    assert_eq!(
        tokenizer.decode_token(TokenId(9999)),
        Err(TokenizerError::InvalidTokenId(TokenId(9999)))
    );
}

#[test]
fn malformed_merge_tables_return_errors() {
    let undefined = vec![MergeRule::new(999, b'a' as u32, 300, 0)];
    assert!(matches!(
        TinyBpeTokenizer::try_new(undefined),
        Err(TokenizerError::InvalidMergeTable(_))
    ));
    let duplicate_rank = vec![
        MergeRule::new(b'a' as u32, b'b' as u32, 300, 0),
        MergeRule::new(b'c' as u32, b'd' as u32, 301, 0),
    ];
    assert!(matches!(
        TinyBpeTokenizer::try_new(duplicate_rank),
        Err(TokenizerError::InvalidMergeTable(_))
    ));
    let special_collision = vec![MergeRule::new(b'a' as u32, b'b' as u32, 1000, 0)];
    assert!(matches!(
        TinyBpeTokenizer::try_new(special_collision),
        Err(TokenizerError::InvalidMergeTable(_))
    ));
}

#[test]
fn engine1_vocabulary_round_trips_supported_text() {
    let tokenizer = TinyLmTokenizer;
    let ids = tokenizer.encode(b"I like Rust").unwrap();
    assert_eq!(ids, vec![TINY_LM_I, TINY_LM_LIKE, TINY_LM_RUST]);
    assert_eq!(tokenizer.decode(&ids).unwrap(), b"I like Rust");
    assert_eq!(tokenizer.vocabulary_size(), 4);
}

#[test]
fn engine1_tokenizer_rejects_out_of_vocabulary_text() {
    assert!(matches!(
        TinyLmTokenizer.encode(b"I dislike Rust"),
        Err(TokenizerError::UnsupportedInput { .. })
    ));
}

#[test]
fn engine1_eos_is_special_and_has_no_text_bytes() {
    assert_eq!(
        TinyLmTokenizer.special_id(SpecialToken::Eos),
        Some(TINY_LM_EOS)
    );
    assert_eq!(
        TinyLmTokenizer.decode_token(TINY_LM_EOS),
        Err(TokenizerError::SpecialTokenHasNoOrdinaryBytes(TINY_LM_EOS))
    );
}
