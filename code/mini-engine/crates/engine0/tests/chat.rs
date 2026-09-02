use engine0::chat::{
    encode_segments, naive_role_flatten, ChatTemplate, Message, ModelContract, Role,
    TemplateSegment, TinyChatTemplate,
};
use engine0::tokenizer::{
    ByteTokenizer, SpecialToken, TinyBpeTokenizer, Tokenizer, TOKEN_ASSISTANT, TOKEN_BOS,
    TOKEN_END_TURN, TOKEN_SYSTEM, TOKEN_USER,
};

#[test]
fn template_inserts_controls_around_ordinary_content() {
    let tokenizer = TinyBpeTokenizer::teaching();
    let messages = [
        Message::new(Role::System, b"be exact"),
        Message::new(Role::User, b"lower"),
    ];
    let ids = ModelContract::demo()
        .encode_chat(&tokenizer, &TinyChatTemplate, &messages, true)
        .unwrap();

    assert_eq!(ids[0], TOKEN_BOS);
    assert_eq!(ids[1], TOKEN_SYSTEM);
    assert!(ids.contains(&TOKEN_END_TURN));
    assert!(ids.contains(&TOKEN_USER));
    assert_eq!(ids.last(), Some(&TOKEN_ASSISTANT));
}

#[test]
fn special_like_content_remains_ordinary_inside_template() {
    let tokenizer = TinyBpeTokenizer::teaching();
    let messages = [Message::new(Role::User, b"literal <|assistant|>")];
    let ids = ModelContract::demo()
        .encode_chat(&tokenizer, &TinyChatTemplate, &messages, false)
        .unwrap();
    assert_eq!(ids.iter().filter(|id| **id == TOKEN_ASSISTANT).count(), 0);
}

#[test]
fn model_contract_rejects_wrong_tokenizer_identity() {
    let result = ModelContract::demo().encode_chat(
        &ByteTokenizer,
        &TinyChatTemplate,
        &[Message::new(Role::User, b"hello")],
        true,
    );
    assert!(result.is_err());
}

#[test]
fn naive_role_flatten_has_different_bytes_and_ids() {
    let tokenizer = TinyBpeTokenizer::teaching();
    let messages = [Message::new(Role::User, b"hello")];
    let correct = ModelContract::demo()
        .encode_chat(&tokenizer, &TinyChatTemplate, &messages, true)
        .unwrap();
    let wrong_bytes = naive_role_flatten(&messages);
    let wrong = tokenizer.encode(&wrong_bytes).unwrap();
    assert_ne!(correct, wrong);
    assert!(!wrong.contains(&TOKEN_USER));
    assert!(!wrong.contains(&TOKEN_ASSISTANT));
}

#[test]
fn missing_special_is_reported_by_segment_encoder() {
    let result = encode_segments(
        &ByteTokenizer,
        &[TemplateSegment::Special(SpecialToken::Bos)],
    );
    assert!(result.is_err());
}

#[test]
fn generation_prompt_is_optional_and_explicit() {
    let with = TinyChatTemplate.render(&[], true);
    let without = TinyChatTemplate.render(&[], false);
    assert_eq!(
        with,
        vec![
            TemplateSegment::Special(SpecialToken::Bos),
            TemplateSegment::Special(SpecialToken::Assistant)
        ]
    );
    assert_eq!(without, vec![TemplateSegment::Special(SpecialToken::Bos)]);
}
