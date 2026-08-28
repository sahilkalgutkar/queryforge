//! Turns SQL text into tokens.
//!
//! Written by hand rather than generated. A SQL lexer is small enough that the
//! interesting decisions — where a keyword stops being a keyword, how a quoted
//! identifier differs from a string literal, what a bare `.` means next to a
//! digit — are worth making explicitly.

use qf_common::{Error, Result};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A bare word that matched no keyword, or a `"quoted identifier"`.
    Ident(String),
    /// A keyword, stored upper-cased so comparisons need no `eq_ignore_case`.
    Keyword(String),
    Int(i64),
    Float(f64),
    /// A `'single quoted'` string literal.
    Str(String),

    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,

    Comma,
    Dot,
    LParen,
    RParen,
    Semicolon,

    /// End of input. Emitted once so the parser never indexes past the end.
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "{s}"),
            Token::Keyword(s) => write!(f, "{s}"),
            Token::Int(i) => write!(f, "{i}"),
            Token::Float(x) => write!(f, "{x}"),
            Token::Str(s) => write!(f, "'{s}'"),
            Token::Plus => f.write_str("+"),
            Token::Minus => f.write_str("-"),
            Token::Star => f.write_str("*"),
            Token::Slash => f.write_str("/"),
            Token::Percent => f.write_str("%"),
            Token::Eq => f.write_str("="),
            Token::NotEq => f.write_str("<>"),
            Token::Lt => f.write_str("<"),
            Token::LtEq => f.write_str("<="),
            Token::Gt => f.write_str(">"),
            Token::GtEq => f.write_str(">="),
            Token::Comma => f.write_str(","),
            Token::Dot => f.write_str("."),
            Token::LParen => f.write_str("("),
            Token::RParen => f.write_str(")"),
            Token::Semicolon => f.write_str(";"),
            Token::Eof => f.write_str("end of input"),
        }
    }
}

/// A token plus where it started, so errors can point at the offending text.
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned {
    pub token: Token,
    pub offset: usize,
}

const KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "LIMIT", "OFFSET", "AS", "AND",
    "OR", "NOT", "NULL", "IS", "IN", "BETWEEN", "LIKE", "DISTINCT", "JOIN", "INNER", "LEFT",
    "RIGHT", "FULL", "OUTER", "CROSS", "ON", "ASC", "DESC", "CASE", "WHEN", "THEN", "ELSE", "END",
    "CAST", "TRUE", "FALSE", "CREATE", "TABLE", "INSERT", "INTO", "VALUES", "DROP", "EXPLAIN",
    "ANALYZE", "NULLS", "FIRST", "LAST", "COPY",
];

pub fn is_keyword(word: &str) -> bool {
    let upper = word.to_ascii_uppercase();
    KEYWORDS.contains(&upper.as_str())
}

pub struct Lexer<'a> {
    src: &'a [u8],
    text: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(text: &'a str) -> Self {
        Lexer {
            src: text.as_bytes(),
            text,
            pos: 0,
        }
    }

    /// Tokenises the whole input, always ending with exactly one `Eof`.
    pub fn tokenize(mut self) -> Result<Vec<Spanned>> {
        let mut out = Vec::new();
        loop {
            let t = self.next_token()?;
            let done = t.token == Token::Eof;
            out.push(t);
            if done {
                return Ok(out);
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek_at(&self, n: usize) -> Option<u8> {
        self.src.get(self.pos + n).copied()
    }

    fn skip_trivia(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_ascii_whitespace() => self.pos += 1,
                // `-- line comment`
                Some(b'-') if self.peek_at(1) == Some(b'-') => {
                    while let Some(c) = self.peek() {
                        self.pos += 1;
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                _ => return,
            }
        }
    }

    fn next_token(&mut self) -> Result<Spanned> {
        self.skip_trivia();
        let offset = self.pos;
        let Some(c) = self.peek() else {
            return Ok(Spanned {
                token: Token::Eof,
                offset,
            });
        };

        let token = match c {
            b'0'..=b'9' => self.number()?,
            b'\'' => self.string()?,
            b'"' => self.quoted_ident()?,
            c if c == b'_' || c.is_ascii_alphabetic() => self.word(),
            b'.' if self.peek_at(1).is_some_and(|d| d.is_ascii_digit()) => self.number()?,
            _ => self.operator()?,
        };
        Ok(Spanned { token, offset })
    }

    fn word(&mut self) -> Token {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == b'_' || c.is_ascii_alphanumeric() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word = &self.text[start..self.pos];
        if is_keyword(word) {
            Token::Keyword(word.to_ascii_uppercase())
        } else {
            Token::Ident(word.to_string())
        }
    }

    fn number(&mut self) -> Result<Token> {
        let start = self.pos;
        let mut is_float = false;
        while let Some(c) = self.peek() {
            match c {
                b'0'..=b'9' => self.pos += 1,
                b'.' if !is_float => {
                    is_float = true;
                    self.pos += 1;
                }
                b'e' | b'E' => {
                    // Only an exponent if a digit or sign actually follows,
                    // so `3e` stays a number followed by an identifier.
                    let next = self.peek_at(1);
                    let after = self.peek_at(2);
                    let exponent = matches!(next, Some(d) if d.is_ascii_digit())
                        || (matches!(next, Some(b'+' | b'-'))
                            && matches!(after, Some(d) if d.is_ascii_digit()));
                    if !exponent {
                        break;
                    }
                    is_float = true;
                    self.pos += 2;
                }
                _ => break,
            }
        }
        let text = &self.text[start..self.pos];
        if is_float {
            text.parse::<f64>()
                .map(Token::Float)
                .map_err(|_| Error::parse(format!("`{text}` is not a valid number")))
        } else {
            match text.parse::<i64>() {
                Ok(i) => Ok(Token::Int(i)),
                // Too big for i64 rather than malformed: keep it as a float
                // instead of refusing the query.
                Err(_) => text
                    .parse::<f64>()
                    .map(Token::Float)
                    .map_err(|_| Error::parse(format!("`{text}` is not a valid number"))),
            }
        }
    }

    fn string(&mut self) -> Result<Token> {
        let start = self.pos;
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(Error::parse(format!(
                        "unterminated string literal starting at offset {start}"
                    )))
                }
                Some(b'\'') => {
                    // '' inside a literal is an escaped quote.
                    if self.peek_at(1) == Some(b'\'') {
                        out.push('\'');
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                        return Ok(Token::Str(out));
                    }
                }
                Some(_) => {
                    let ch = self.text[self.pos..].chars().next().unwrap();
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn quoted_ident(&mut self) -> Result<Token> {
        let start = self.pos;
        self.pos += 1;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => {
                    return Err(Error::parse(format!(
                        "unterminated quoted identifier starting at offset {start}"
                    )))
                }
                Some(b'"') => {
                    if self.peek_at(1) == Some(b'"') {
                        out.push('"');
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                        return Ok(Token::Ident(out));
                    }
                }
                Some(_) => {
                    let ch = self.text[self.pos..].chars().next().unwrap();
                    out.push(ch);
                    self.pos += ch.len_utf8();
                }
            }
        }
    }

    fn operator(&mut self) -> Result<Token> {
        let two: Option<&str> = self.text.get(self.pos..self.pos + 2);
        if let Some(op) = two {
            let token = match op {
                "<=" => Some(Token::LtEq),
                ">=" => Some(Token::GtEq),
                "<>" | "!=" => Some(Token::NotEq),
                "||" => None, // reserved for concatenation; not supported yet
                _ => None,
            };
            if let Some(t) = token {
                self.pos += 2;
                return Ok(t);
            }
        }
        let c = self.peek().unwrap();
        let token = match c {
            b'+' => Token::Plus,
            b'-' => Token::Minus,
            b'*' => Token::Star,
            b'/' => Token::Slash,
            b'%' => Token::Percent,
            b'=' => Token::Eq,
            b'<' => Token::Lt,
            b'>' => Token::Gt,
            b',' => Token::Comma,
            b'.' => Token::Dot,
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b';' => Token::Semicolon,
            other => {
                return Err(Error::parse(format!(
                    "unexpected character `{}` at offset {}",
                    other as char, self.pos
                )))
            }
        };
        self.pos += 1;
        Ok(token)
    }
}

pub fn tokenize(sql: &str) -> Result<Vec<Spanned>> {
    Lexer::new(sql).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(sql: &str) -> Vec<Token> {
        tokenize(sql)
            .unwrap()
            .into_iter()
            .map(|s| s.token)
            .collect()
    }

    #[test]
    fn a_simple_select_lexes_into_keywords_identifiers_and_punctuation() {
        assert_eq!(
            toks("SELECT a, b FROM t;"),
            vec![
                Token::Keyword("SELECT".into()),
                Token::Ident("a".into()),
                Token::Comma,
                Token::Ident("b".into()),
                Token::Keyword("FROM".into()),
                Token::Ident("t".into()),
                Token::Semicolon,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn keywords_are_normalised_to_upper_case_whatever_was_typed() {
        assert_eq!(
            toks("select Where FROM"),
            vec![
                Token::Keyword("SELECT".into()),
                Token::Keyword("WHERE".into()),
                Token::Keyword("FROM".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn identifiers_keep_the_case_they_were_written_in() {
        assert_eq!(toks("MyTable")[0], Token::Ident("MyTable".into()));
    }

    #[test]
    fn a_quoted_identifier_may_hold_a_keyword_or_a_space() {
        assert_eq!(
            toks(r#""select", "two words", "a""b""#),
            vec![
                Token::Ident("select".into()),
                Token::Comma,
                Token::Ident("two words".into()),
                Token::Comma,
                Token::Ident("a\"b".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn numbers_split_into_integers_and_floats() {
        assert_eq!(
            toks("1 2.5 .5 1e3 1.5e-2"),
            vec![
                Token::Int(1),
                Token::Float(2.5),
                Token::Float(0.5),
                Token::Float(1000.0),
                Token::Float(0.015),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn a_trailing_e_is_not_swallowed_into_the_number() {
        // `3e` is 3 followed by the identifier `e`, not a malformed float.
        assert_eq!(
            toks("3e"),
            vec![Token::Int(3), Token::Ident("e".into()), Token::Eof]
        );
    }

    #[test]
    fn an_integer_too_large_for_i64_becomes_a_float_rather_than_an_error() {
        assert!(matches!(toks("99999999999999999999")[0], Token::Float(_)));
    }

    #[test]
    fn string_literals_handle_doubled_quotes_and_unicode() {
        assert_eq!(
            toks("'it''s', 'naïve'"),
            vec![
                Token::Str("it's".into()),
                Token::Comma,
                Token::Str("naïve".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn comparison_operators_lex_as_single_tokens() {
        assert_eq!(
            toks("< <= > >= = <> !="),
            vec![
                Token::Lt,
                Token::LtEq,
                Token::Gt,
                Token::GtEq,
                Token::Eq,
                Token::NotEq,
                Token::NotEq,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn arithmetic_operators_lex_as_single_tokens() {
        assert_eq!(
            toks("+ - * / % . ( )"),
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::Percent,
                Token::Dot,
                Token::LParen,
                Token::RParen,
                Token::Eof,
            ]
        );
    }

    #[test]
    fn line_comments_and_whitespace_are_skipped() {
        assert_eq!(
            toks("SELECT -- everything\n  a\n-- trailing comment"),
            vec![
                Token::Keyword("SELECT".into()),
                Token::Ident("a".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn a_minus_that_is_not_a_comment_stays_an_operator() {
        assert_eq!(
            toks("a - b"),
            vec![
                Token::Ident("a".into()),
                Token::Minus,
                Token::Ident("b".into()),
                Token::Eof,
            ]
        );
    }

    #[test]
    fn unterminated_literals_are_reported_with_where_they_started() {
        let err = tokenize("SELECT 'oops").unwrap_err();
        assert!(err.to_string().contains("unterminated string"));
        assert!(err.to_string().contains("offset 7"));
        assert!(tokenize(r#"SELECT "oops"#)
            .unwrap_err()
            .to_string()
            .contains("unterminated quoted identifier"));
    }

    #[test]
    fn an_unexpected_character_is_reported_with_its_offset() {
        let err = tokenize("SELECT a # b").unwrap_err();
        assert!(err.to_string().contains('#'));
        assert!(err.to_string().contains("offset 9"));
    }

    #[test]
    fn the_double_pipe_operator_is_refused_rather_than_silently_split() {
        assert!(tokenize("a || b").is_err());
    }

    #[test]
    fn spans_point_at_where_each_token_started() {
        let spans = tokenize("SELECT  a").unwrap();
        assert_eq!(spans[0].offset, 0);
        assert_eq!(spans[1].offset, 8);
    }

    #[test]
    fn empty_input_lexes_to_just_eof() {
        assert_eq!(toks(""), vec![Token::Eof]);
        assert_eq!(toks("   -- nothing here\n  "), vec![Token::Eof]);
    }

    #[test]
    fn tokens_render_back_to_something_readable_in_errors() {
        assert_eq!(Token::Keyword("SELECT".into()).to_string(), "SELECT");
        assert_eq!(Token::Str("x".into()).to_string(), "'x'");
        assert_eq!(Token::NotEq.to_string(), "<>");
        assert_eq!(Token::Eof.to_string(), "end of input");
        assert_eq!(Token::Float(1.5).to_string(), "1.5");
        assert_eq!(Token::Int(2).to_string(), "2");
        assert_eq!(Token::Ident("c".into()).to_string(), "c");
        for t in [
            Token::Plus,
            Token::Minus,
            Token::Star,
            Token::Slash,
            Token::Percent,
            Token::Eq,
            Token::Lt,
            Token::LtEq,
            Token::Gt,
            Token::GtEq,
            Token::Comma,
            Token::Dot,
            Token::LParen,
            Token::RParen,
            Token::Semicolon,
        ] {
            assert!(!t.to_string().is_empty());
        }
    }

    #[test]
    fn the_keyword_list_is_queryable() {
        assert!(is_keyword("select"));
        assert!(is_keyword("JOIN"));
        assert!(!is_keyword("customers"));
    }
}
