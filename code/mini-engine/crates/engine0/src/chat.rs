use std::fmt;

use crate::tokenizer::{SpecialToken, TokenId, Tokenizer, TokenizerError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    fn special(self) -> SpecialToken {
        match self {
            Self::System => SpecialToken::System,
            Self::User => SpecialToken::User,
            Self::Assistant => SpecialToken::Assistant,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Message<'a> {
    pub role: Role,
    pub content: &'a [u8],
}

impl<'a> Message<'a> {
    pub const fn new(role: Role, content: &'a [u8]) -> Self {
        Self { role, content }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateSegment {
    Text(Vec<u8>),
    Special(SpecialToken),
}

pub trait ChatTemplate {
    fn identity(&self) -> &'static str;
    fn render(&self, messages: &[Message<'_>], add_generation_prompt: bool)
        -> Vec<TemplateSegment>;
}

/// A tiny structured template: BOS, then role/content/end-turn for each
/// message, followed by an assistant role when generation should begin.
#[derive(Debug, Default, Clone, Copy)]
pub struct TinyChatTemplate;

impl ChatTemplate for TinyChatTemplate {
    fn identity(&self) -> &'static str {
        "tiny-chat-v1"
    }

    fn render(
        &self,
        messages: &[Message<'_>],
        add_generation_prompt: bool,
    ) -> Vec<TemplateSegment> {
        let mut output = vec![TemplateSegment::Special(SpecialToken::Bos)];
        for message in messages {
            output.push(TemplateSegment::Special(message.role.special()));
            output.push(TemplateSegment::Text(message.content.to_vec()));
            output.push(TemplateSegment::Special(SpecialToken::EndTurn));
        }
        if add_generation_prompt {
            output.push(TemplateSegment::Special(SpecialToken::Assistant));
        }
        output
    }
}

pub fn encode_segments(
    tokenizer: &impl Tokenizer,
    segments: &[TemplateSegment],
) -> Result<Vec<TokenId>, TokenizerError> {
    let mut output = Vec::new();
    for segment in segments {
        match segment {
            TemplateSegment::Text(bytes) => output.extend(tokenizer.encode(bytes)?),
            TemplateSegment::Special(special) => {
                output.push(
                    tokenizer
                        .special_id(*special)
                        .ok_or(TokenizerError::MissingSpecialToken(*special))?,
                );
            }
        }
    }
    Ok(output)
}

/// Deliberately wrong for the teaching model. It is useful only for comparing
/// bytes and IDs with the structured template; ENGINE-0 cannot measure quality.
pub fn naive_role_flatten(messages: &[Message<'_>]) -> Vec<u8> {
    let mut output = Vec::new();
    for message in messages {
        output.extend_from_slice(message.role.name().as_bytes());
        output.extend_from_slice(b": ");
        output.extend_from_slice(message.content);
        output.push(b'\n');
    }
    output
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelContract {
    pub model: &'static str,
    pub tokenizer: &'static str,
    pub tokenizer_revision: &'static str,
    pub chat_template: &'static str,
}

impl ModelContract {
    pub const fn demo() -> Self {
        Self {
            model: "engine0-demo-model",
            tokenizer: "tiny-byte-bpe",
            tokenizer_revision: "chapter-02-v1",
            chat_template: "tiny-chat-v1",
        }
    }

    pub fn encode_chat(
        &self,
        tokenizer: &impl Tokenizer,
        template: &impl ChatTemplate,
        messages: &[Message<'_>],
        add_generation_prompt: bool,
    ) -> Result<Vec<TokenId>, ContractError> {
        let identity = tokenizer.identity();
        if identity.name != self.tokenizer || identity.revision != self.tokenizer_revision {
            return Err(ContractError::TokenizerMismatch {
                expected: format!("{}@{}", self.tokenizer, self.tokenizer_revision),
                actual: format!("{}@{}", identity.name, identity.revision),
            });
        }
        if template.identity() != self.chat_template {
            return Err(ContractError::TemplateMismatch {
                expected: self.chat_template.to_string(),
                actual: template.identity().to_string(),
            });
        }
        Ok(encode_segments(
            tokenizer,
            &template.render(messages, add_generation_prompt),
        )?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    TokenizerMismatch { expected: String, actual: String },
    TemplateMismatch { expected: String, actual: String },
    Tokenizer(TokenizerError),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenizerMismatch { expected, actual } => {
                write!(f, "tokenizer mismatch: expected {expected}, got {actual}")
            }
            Self::TemplateMismatch { expected, actual } => {
                write!(
                    f,
                    "chat-template mismatch: expected {expected}, got {actual}"
                )
            }
            Self::Tokenizer(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ContractError {}

impl From<TokenizerError> for ContractError {
    fn from(value: TokenizerError) -> Self {
        Self::Tokenizer(value)
    }
}
