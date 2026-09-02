use std::fmt;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn pending_bytes(&self) -> &[u8] {
        &self.pending
    }

    /// Append token bytes and emit the longest complete, valid UTF-8 prefix
    /// when the remainder is only incomplete. A definite error preserves the
    /// buffer and emits nothing from that append.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Option<String>, Utf8StreamError> {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let output = (!text.is_empty()).then(|| text.to_owned());
                self.pending.clear();
                Ok(output)
            }
            Err(error) if error.error_len().is_none() => {
                let valid_up_to = error.valid_up_to();
                if self.pending.len().saturating_sub(valid_up_to) > 3 {
                    return Err(Utf8StreamError::InvalidSequence {
                        valid_up_to,
                        error_len: None,
                        bytes: self.pending.clone(),
                    });
                }
                if valid_up_to == 0 {
                    return Ok(None);
                }
                let prefix = std::str::from_utf8(&self.pending[..valid_up_to])
                    .map_err(|_| Utf8StreamError::InvalidSequence {
                        valid_up_to,
                        error_len: None,
                        bytes: self.pending.clone(),
                    })?
                    .to_owned();
                self.pending.drain(..valid_up_to);
                Ok(Some(prefix))
            }
            Err(error) => Err(Utf8StreamError::InvalidSequence {
                valid_up_to: error.valid_up_to(),
                error_len: error.error_len(),
                bytes: self.pending.clone(),
            }),
        }
    }

    pub fn finish(&self) -> Result<(), Utf8StreamError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(Utf8StreamError::IncompleteSequence {
                bytes: self.pending.clone(),
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Utf8StreamError {
    InvalidSequence {
        valid_up_to: usize,
        error_len: Option<usize>,
        bytes: Vec<u8>,
    },
    IncompleteSequence {
        bytes: Vec<u8>,
    },
}

impl fmt::Display for Utf8StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSequence {
                valid_up_to,
                error_len,
                bytes,
            } => write!(
                f,
                "invalid UTF-8 after byte {valid_up_to} (error length {error_len:?}, buffered {bytes:02x?})"
            ),
            Self::IncompleteSequence { bytes } => {
                write!(f, "incomplete UTF-8 at terminal (buffered {bytes:02x?})")
            }
        }
    }
}

impl std::error::Error for Utf8StreamError {}
