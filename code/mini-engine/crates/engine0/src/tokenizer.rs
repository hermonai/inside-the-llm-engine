use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// A model-vocabulary identity. The numeric value is meaningful only with the
/// tokenizer artifact and configuration that define it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TokenId(pub u32);

impl fmt::Display for TokenId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerIdentity {
    pub name: &'static str,
    pub revision: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpecialToken {
    Bos,
    Eos,
    Pad,
    Unk,
    System,
    User,
    Assistant,
    EndTurn,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    InvalidTokenId(TokenId),
    SpecialTokenHasNoOrdinaryBytes(TokenId),
    MissingSpecialToken(SpecialToken),
    InvalidMergeTable(String),
    UnsupportedInput { offset: usize },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTokenId(id) => write!(f, "token id {id} is not in this vocabulary"),
            Self::SpecialTokenHasNoOrdinaryBytes(id) => {
                write!(f, "special token id {id} has no ordinary text bytes")
            }
            Self::MissingSpecialToken(token) => {
                write!(f, "tokenizer has no configured {token:?} token")
            }
            Self::InvalidMergeTable(message) => write!(f, "invalid BPE merge table: {message}"),
            Self::UnsupportedInput { offset } => write!(
                f,
                "input at byte offset {offset} has no token in this tiny vocabulary"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Byte-oriented tokenizer boundary. Special insertion is deliberately absent
/// from `encode`: ordinary user bytes cannot become control identities merely
/// by spelling a marker-looking string.
pub trait Tokenizer {
    fn identity(&self) -> TokenizerIdentity;
    fn vocabulary_size(&self) -> usize;
    fn encode(&self, input: &[u8]) -> Result<Vec<TokenId>, TokenizerError>;
    fn decode_token(&self, id: TokenId) -> Result<&[u8], TokenizerError>;
    fn special_id(&self, token: SpecialToken) -> Option<TokenId>;

    fn decode(&self, ids: &[TokenId]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::new();
        for &id in ids {
            output.extend_from_slice(self.decode_token(id)?);
        }
        Ok(output)
    }
}

/// Independent oracle: each byte maps to the token ID with the same value.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn identity(&self) -> TokenizerIdentity {
        TokenizerIdentity {
            name: "byte-oracle",
            revision: "chapter-02-v1",
        }
    }

    fn vocabulary_size(&self) -> usize {
        256
    }

    fn encode(&self, input: &[u8]) -> Result<Vec<TokenId>, TokenizerError> {
        Ok(input.iter().map(|byte| TokenId(u32::from(*byte))).collect())
    }

    fn decode_token(&self, id: TokenId) -> Result<&[u8], TokenizerError> {
        static BYTES: [[u8; 1]; 256] = make_byte_table();
        BYTES
            .get(id.0 as usize)
            .map(|byte| byte.as_slice())
            .ok_or(TokenizerError::InvalidTokenId(id))
    }

    fn special_id(&self, _token: SpecialToken) -> Option<TokenId> {
        None
    }
}

const fn make_byte_table() -> [[u8; 1]; 256] {
    let mut table = [[0u8; 1]; 256];
    let mut index = 0;
    while index < 256 {
        table[index][0] = index as u8;
        index += 1;
    }
    table
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MergeRule {
    pub left: TokenId,
    pub right: TokenId,
    pub result: TokenId,
    pub rank: usize,
}

impl MergeRule {
    pub const fn new(left: u32, right: u32, result: u32, rank: usize) -> Self {
        Self {
            left: TokenId(left),
            right: TokenId(right),
            result: TokenId(result),
            rank,
        }
    }
}

pub const TOKEN_LO: TokenId = TokenId(256);
pub const TOKEN_LOW: TokenId = TokenId(257);
pub const TOKEN_ER: TokenId = TokenId(258);
pub const TOKEN_LOWER: TokenId = TokenId(259);
pub const TOKEN_BL: TokenId = TokenId(260);
pub const TOKEN_BLU: TokenId = TokenId(261);
pub const TOKEN_BLUE: TokenId = TokenId(262);
pub const TOKEN_GR: TokenId = TokenId(263);
pub const TOKEN_GRE: TokenId = TokenId(264);
pub const TOKEN_GREE: TokenId = TokenId(265);
pub const TOKEN_GREEN: TokenId = TokenId(266);

pub const TOKEN_BOS: TokenId = TokenId(1000);
pub const TOKEN_EOS: TokenId = TokenId(1001);
pub const TOKEN_PAD: TokenId = TokenId(1002);
pub const TOKEN_UNK: TokenId = TokenId(1003);
pub const TOKEN_SYSTEM: TokenId = TokenId(1004);
pub const TOKEN_USER: TokenId = TokenId(1005);
pub const TOKEN_ASSISTANT: TokenId = TokenId(1006);
pub const TOKEN_END_TURN: TokenId = TokenId(1007);

/// A deliberately small byte-fallback BPE. Its quadratic merge scan is easy to
/// inspect and intentionally not presented as a production optimization.
#[derive(Debug, Clone)]
pub struct TinyBpeTokenizer {
    merges: Vec<MergeRule>,
    pieces: BTreeMap<TokenId, Vec<u8>>,
    specials: BTreeMap<SpecialToken, TokenId>,
}

impl Default for TinyBpeTokenizer {
    fn default() -> Self {
        Self::teaching()
    }
}

impl TinyBpeTokenizer {
    pub fn teaching() -> Self {
        let rules = vec![
            MergeRule::new(b'l' as u32, b'o' as u32, TOKEN_LO.0, 0),
            MergeRule::new(TOKEN_LO.0, b'w' as u32, TOKEN_LOW.0, 1),
            MergeRule::new(b'e' as u32, b'r' as u32, TOKEN_ER.0, 2),
            MergeRule::new(TOKEN_LOW.0, TOKEN_ER.0, TOKEN_LOWER.0, 3),
            MergeRule::new(b'b' as u32, b'l' as u32, TOKEN_BL.0, 4),
            MergeRule::new(TOKEN_BL.0, b'u' as u32, TOKEN_BLU.0, 5),
            MergeRule::new(TOKEN_BLU.0, b'e' as u32, TOKEN_BLUE.0, 6),
            MergeRule::new(b'g' as u32, b'r' as u32, TOKEN_GR.0, 7),
            MergeRule::new(TOKEN_GR.0, b'e' as u32, TOKEN_GRE.0, 8),
            MergeRule::new(TOKEN_GRE.0, b'e' as u32, TOKEN_GREE.0, 9),
            MergeRule::new(TOKEN_GREE.0, b'n' as u32, TOKEN_GREEN.0, 10),
        ];
        match Self::try_new(rules) {
            Ok(tokenizer) => tokenizer,
            Err(error) => panic!("built-in teaching merge table must be valid: {error}"),
        }
    }

    pub fn try_new(merges: Vec<MergeRule>) -> Result<Self, TokenizerError> {
        let mut pieces: BTreeMap<TokenId, Vec<u8>> = (0u32..=255)
            .map(|value| (TokenId(value), vec![value as u8]))
            .collect();
        let mut pairs = HashSet::new();
        let mut ranks = HashSet::new();
        let mut results = HashSet::new();

        let mut ordered = merges;
        ordered.sort_by_key(|rule| rule.rank);
        for rule in &ordered {
            if !pairs.insert((rule.left, rule.right)) {
                return Err(TokenizerError::InvalidMergeTable(format!(
                    "duplicate pair ({}, {})",
                    rule.left, rule.right
                )));
            }
            if !ranks.insert(rule.rank) {
                return Err(TokenizerError::InvalidMergeTable(format!(
                    "duplicate rank {}",
                    rule.rank
                )));
            }
            if rule.result.0 <= 255
                || (TOKEN_BOS.0..=TOKEN_END_TURN.0).contains(&rule.result.0)
                || !results.insert(rule.result)
            {
                return Err(TokenizerError::InvalidMergeTable(format!(
                    "result id {} is reserved, special, or duplicated",
                    rule.result
                )));
            }
            let left = pieces.get(&rule.left).ok_or_else(|| {
                TokenizerError::InvalidMergeTable(format!(
                    "left id {} is undefined at rank {}",
                    rule.left, rule.rank
                ))
            })?;
            let right = pieces.get(&rule.right).ok_or_else(|| {
                TokenizerError::InvalidMergeTable(format!(
                    "right id {} is undefined at rank {}",
                    rule.right, rule.rank
                ))
            })?;
            let mut piece = Vec::with_capacity(left.len() + right.len());
            piece.extend_from_slice(left);
            piece.extend_from_slice(right);
            pieces.insert(rule.result, piece);
        }

        let specials = [
            (SpecialToken::Bos, TOKEN_BOS),
            (SpecialToken::Eos, TOKEN_EOS),
            (SpecialToken::Pad, TOKEN_PAD),
            (SpecialToken::Unk, TOKEN_UNK),
            (SpecialToken::System, TOKEN_SYSTEM),
            (SpecialToken::User, TOKEN_USER),
            (SpecialToken::Assistant, TOKEN_ASSISTANT),
            (SpecialToken::EndTurn, TOKEN_END_TURN),
        ]
        .into_iter()
        .collect();

        Ok(Self {
            merges: ordered,
            pieces,
            specials,
        })
    }

    pub fn merge_rules(&self) -> &[MergeRule] {
        &self.merges
    }
}

impl Tokenizer for TinyBpeTokenizer {
    fn identity(&self) -> TokenizerIdentity {
        TokenizerIdentity {
            name: "tiny-byte-bpe",
            revision: "chapter-02-v1",
        }
    }

    fn vocabulary_size(&self) -> usize {
        // The teaching vocabulary deliberately reserves sparse special-token
        // IDs through 1007. A compatible logit vector therefore needs a row
        // for every numeric identity in 0..=1007, including unused gaps.
        TOKEN_END_TURN.0 as usize + 1
    }

    fn encode(&self, input: &[u8]) -> Result<Vec<TokenId>, TokenizerError> {
        let mut tokens: Vec<TokenId> = input.iter().map(|byte| TokenId(u32::from(*byte))).collect();
        if tokens.len() < 2 {
            return Ok(tokens);
        }

        let by_pair: HashMap<(TokenId, TokenId), MergeRule> = self
            .merges
            .iter()
            .copied()
            .map(|rule| ((rule.left, rule.right), rule))
            .collect();

        loop {
            let mut best: Option<(usize, MergeRule)> = None;
            for index in 0..tokens.len().saturating_sub(1) {
                if let Some(&rule) = by_pair.get(&(tokens[index], tokens[index + 1])) {
                    let replace = best
                        .as_ref()
                        .map(|(_, current)| rule.rank < current.rank)
                        .unwrap_or(true);
                    if replace {
                        best = Some((index, rule));
                    }
                }
            }
            let Some((index, rule)) = best else {
                break;
            };
            tokens.splice(index..=index + 1, [rule.result]);
        }
        Ok(tokens)
    }

    fn decode_token(&self, id: TokenId) -> Result<&[u8], TokenizerError> {
        if self.specials.values().any(|special_id| *special_id == id) {
            return Err(TokenizerError::SpecialTokenHasNoOrdinaryBytes(id));
        }
        self.pieces
            .get(&id)
            .map(Vec::as_slice)
            .ok_or(TokenizerError::InvalidTokenId(id))
    }

    fn special_id(&self, token: SpecialToken) -> Option<TokenId> {
        self.specials.get(&token).copied()
    }
}

pub const TINY_LM_EOS: TokenId = TokenId(0);
pub const TINY_LM_I: TokenId = TokenId(1);
pub const TINY_LM_LIKE: TokenId = TokenId(2);
pub const TINY_LM_RUST: TokenId = TokenId(3);

/// The vocabulary paired with the four-row ENGINE-1 fixture.
///
/// This is intentionally a complete model/tokenizer contract, not a remapping
/// layered over Chapter 2's byte-BPE IDs. Leading spaces belong to the `like`
/// and `Rust` pieces, so ordinary text such as `I like Rust` round-trips while
/// each vocabulary identity still names exactly one embedding/logit row.
#[derive(Debug, Default, Clone, Copy)]
pub struct TinyLmTokenizer;

impl Tokenizer for TinyLmTokenizer {
    fn identity(&self) -> TokenizerIdentity {
        TokenizerIdentity {
            name: "engine1-four-token",
            revision: "chapter-03-v1",
        }
    }

    fn vocabulary_size(&self) -> usize {
        4
    }

    fn encode(&self, input: &[u8]) -> Result<Vec<TokenId>, TokenizerError> {
        let mut remaining = input;
        let mut ids = Vec::new();
        while !remaining.is_empty() {
            let (id, width) = if remaining.starts_with(b" Rust") {
                (TINY_LM_RUST, 5)
            } else if remaining.starts_with(b" like") {
                (TINY_LM_LIKE, 5)
            } else if remaining.starts_with(b"I") {
                (TINY_LM_I, 1)
            } else {
                return Err(TokenizerError::UnsupportedInput {
                    offset: input.len() - remaining.len(),
                });
            };
            ids.push(id);
            remaining = &remaining[width..];
        }
        Ok(ids)
    }

    fn decode_token(&self, id: TokenId) -> Result<&[u8], TokenizerError> {
        match id {
            TINY_LM_I => Ok(b"I"),
            TINY_LM_LIKE => Ok(b" like"),
            TINY_LM_RUST => Ok(b" Rust"),
            TINY_LM_EOS => Err(TokenizerError::SpecialTokenHasNoOrdinaryBytes(id)),
            _ => Err(TokenizerError::InvalidTokenId(id)),
        }
    }

    fn special_id(&self, token: SpecialToken) -> Option<TokenId> {
        (token == SpecialToken::Eos).then_some(TINY_LM_EOS)
    }
}
