//! Recursive descent for statements, precedence climbing for expressions.
//!
//! The whole grammar is small enough to read top to bottom, which is the point:
//! when a query parses into something surprising, the rule that did it is a few
//! screens away rather than inside a generated table.

use crate::ast::*;
use crate::lexer::{tokenize, Spanned, Token};
use qf_common::{DataType, Error, Result, Value};

/// Binding power, loosest first. `NOT` binds tighter than `AND` and looser
/// than any comparison, so `NOT a = b` is `NOT (a = b)`.
const PREC_OR: u8 = 1;
const PREC_AND: u8 = 2;
const PREC_NOT: u8 = 3;
const PREC_COMPARE: u8 = 4;
const PREC_ADD: u8 = 5;
const PREC_MUL: u8 = 6;

pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    pub fn new(sql: &str) -> Result<Parser> {
        Ok(Parser {
            tokens: tokenize(sql)?,
            pos: 0,
        })
    }

    /// Parses every statement in the input, separated by semicolons.
    pub fn parse_statements(sql: &str) -> Result<Vec<Statement>> {
        let mut p = Parser::new(sql)?;
        let mut out = Vec::new();
        loop {
            while p.eat(&Token::Semicolon) {}
            if p.peek() == &Token::Eof {
                return Ok(out);
            }
            out.push(p.parse_statement()?);
            if !matches!(p.peek(), Token::Semicolon | Token::Eof) {
                return Err(p.unexpected("a semicolon or the end of the statement"));
            }
        }
    }

    /// Parses exactly one statement and refuses trailing input.
    pub fn parse_one(sql: &str) -> Result<Statement> {
        let mut stmts = Parser::parse_statements(sql)?;
        match stmts.len() {
            0 => Err(Error::parse("no statement to run")),
            1 => Ok(stmts.remove(0)),
            n => Err(Error::parse(format!(
                "expected a single statement, found {n}"
            ))),
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].token
    }

    fn peek_at(&self, n: usize) -> &Token {
        let i = (self.pos + n).min(self.tokens.len() - 1);
        &self.tokens[i].token
    }

    fn next(&mut self) -> Token {
        let t = self.tokens[self.pos].token.clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, want: &Token) -> bool {
        if self.peek() == want {
            self.next();
            true
        } else {
            false
        }
    }

    fn eat_keyword(&mut self, word: &str) -> bool {
        if matches!(self.peek(), Token::Keyword(k) if k == word) {
            self.next();
            true
        } else {
            false
        }
    }

    fn peek_keyword(&self, word: &str) -> bool {
        matches!(self.peek(), Token::Keyword(k) if k == word)
    }

    fn expect(&mut self, want: &Token) -> Result<()> {
        if self.eat(want) {
            Ok(())
        } else {
            Err(self.unexpected(&format!("`{want}`")))
        }
    }

    fn expect_keyword(&mut self, word: &str) -> Result<()> {
        if self.eat_keyword(word) {
            Ok(())
        } else {
            Err(self.unexpected(&format!("`{word}`")))
        }
    }

    fn unexpected(&self, wanted: &str) -> Error {
        let s = &self.tokens[self.pos];
        Error::parse(format!(
            "expected {wanted}, found `{}` at offset {}",
            s.token, s.offset
        ))
    }

    /// An identifier, or a non-reserved keyword being used as a name.
    fn identifier(&mut self) -> Result<String> {
        match self.peek().clone() {
            Token::Ident(name) => {
                self.next();
                Ok(name)
            }
            _ => Err(self.unexpected("an identifier")),
        }
    }

    // ---- statements ----

    fn parse_statement(&mut self) -> Result<Statement> {
        match self.peek().clone() {
            Token::Keyword(k) => match k.as_str() {
                "SELECT" => Ok(Statement::Query(Box::new(self.parse_query()?))),
                "CREATE" => self.parse_create_table(),
                "INSERT" => self.parse_insert(),
                "COPY" => self.parse_copy(),
                "DROP" => self.parse_drop(),
                "EXPLAIN" => self.parse_explain(),
                _ => Err(self.unexpected("a statement")),
            },
            _ => Err(self.unexpected("a statement")),
        }
    }

    fn parse_create_table(&mut self) -> Result<Statement> {
        self.expect_keyword("CREATE")?;
        self.expect_keyword("TABLE")?;
        let name = self.identifier()?;
        self.expect(&Token::LParen)?;
        let mut columns = Vec::new();
        loop {
            let col = self.identifier()?;
            let type_name = match self.peek().clone() {
                Token::Ident(t) => {
                    self.next();
                    t
                }
                _ => return Err(self.unexpected("a column type")),
            };
            let data_type = DataType::from_sql_name(&type_name)?;
            // `NOT NULL` marks the column non-nullable; everything else is.
            let nullable = !(self.eat_keyword("NOT") && {
                self.expect_keyword("NULL")?;
                true
            });
            columns.push(ColumnDef {
                name: col,
                data_type,
                nullable,
            });
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        self.expect(&Token::RParen)?;
        if columns.is_empty() {
            return Err(Error::parse("a table needs at least one column"));
        }
        Ok(Statement::CreateTable { name, columns })
    }

    fn parse_insert(&mut self) -> Result<Statement> {
        self.expect_keyword("INSERT")?;
        self.expect_keyword("INTO")?;
        let table = self.identifier()?;
        self.expect_keyword("VALUES")?;
        let mut rows = Vec::new();
        loop {
            self.expect(&Token::LParen)?;
            let mut row = Vec::new();
            loop {
                row.push(self.parse_expr(0)?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RParen)?;
            rows.push(row);
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(Statement::Insert { table, rows })
    }

    fn parse_copy(&mut self) -> Result<Statement> {
        self.expect_keyword("COPY")?;
        let table = self.identifier()?;
        self.expect_keyword("FROM")?;
        match self.next() {
            Token::Str(path) => Ok(Statement::Copy { table, path }),
            _ => Err(Error::parse(
                "COPY needs a quoted file path, as in COPY t FROM 'data.csv'",
            )),
        }
    }

    fn parse_drop(&mut self) -> Result<Statement> {
        self.expect_keyword("DROP")?;
        self.expect_keyword("TABLE")?;
        Ok(Statement::DropTable {
            name: self.identifier()?,
        })
    }

    fn parse_explain(&mut self) -> Result<Statement> {
        self.expect_keyword("EXPLAIN")?;
        let analyze = self.eat_keyword("ANALYZE");
        Ok(Statement::Explain {
            analyze,
            query: Box::new(self.parse_query()?),
        })
    }

    // ---- queries ----

    fn parse_query(&mut self) -> Result<Query> {
        self.expect_keyword("SELECT")?;
        let distinct = self.eat_keyword("DISTINCT");
        let projection = self.parse_projection()?;

        let from = if self.eat_keyword("FROM") {
            Some(self.parse_table_ref()?)
        } else {
            None
        };

        let selection = if self.eat_keyword("WHERE") {
            Some(self.parse_expr(0)?)
        } else {
            None
        };

        let mut group_by = Vec::new();
        if self.eat_keyword("GROUP") {
            self.expect_keyword("BY")?;
            loop {
                group_by.push(self.parse_expr(0)?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }

        let having = if self.eat_keyword("HAVING") {
            Some(self.parse_expr(0)?)
        } else {
            None
        };

        let mut order_by = Vec::new();
        if self.eat_keyword("ORDER") {
            self.expect_keyword("BY")?;
            loop {
                let expr = self.parse_expr(0)?;
                let ascending = if self.eat_keyword("DESC") {
                    false
                } else {
                    self.eat_keyword("ASC");
                    true
                };
                // Postgres' default: nulls sort last ascending, first
                // descending, so `ORDER BY x DESC` puts real values on top.
                let mut nulls_first = !ascending;
                if self.eat_keyword("NULLS") {
                    if self.eat_keyword("FIRST") {
                        nulls_first = true;
                    } else {
                        self.expect_keyword("LAST")?;
                        nulls_first = false;
                    }
                }
                order_by.push(OrderByExpr {
                    expr,
                    ascending,
                    nulls_first,
                });
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
        }

        let limit = if self.eat_keyword("LIMIT") {
            Some(self.parse_count("LIMIT")?)
        } else {
            None
        };
        let offset = if self.eat_keyword("OFFSET") {
            Some(self.parse_count("OFFSET")?)
        } else {
            None
        };

        if having.is_some() && group_by.is_empty() {
            // A bare HAVING is legal in some dialects over the implicit single
            // group, but it is far more often a WHERE that was mistyped.
            let projects_aggregate = projection.iter().any(|item| match item {
                SelectItem::Expr { expr, .. } => expr.contains_aggregate(),
                _ => false,
            });
            if !projects_aggregate {
                return Err(Error::parse(
                    "HAVING without GROUP BY and without an aggregate — did you mean WHERE?",
                ));
            }
        }

        Ok(Query {
            distinct,
            projection,
            from,
            selection,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_count(&mut self, clause: &str) -> Result<usize> {
        match self.next() {
            Token::Int(i) if i >= 0 => Ok(i as usize),
            other => Err(Error::parse(format!(
                "{clause} needs a non-negative integer, found `{other}`"
            ))),
        }
    }

    fn parse_projection(&mut self) -> Result<Vec<SelectItem>> {
        let mut items = Vec::new();
        loop {
            items.push(self.parse_select_item()?);
            if !self.eat(&Token::Comma) {
                break;
            }
        }
        Ok(items)
    }

    fn parse_select_item(&mut self) -> Result<SelectItem> {
        if self.eat(&Token::Star) {
            return Ok(SelectItem::Wildcard);
        }
        // `t.*`
        if let (Token::Ident(name), Token::Dot, Token::Star) = (
            self.peek().clone(),
            self.peek_at(1).clone(),
            self.peek_at(2),
        ) {
            self.next();
            self.next();
            self.next();
            return Ok(SelectItem::QualifiedWildcard(name));
        }
        let expr = self.parse_expr(0)?;
        let alias = if self.eat_keyword("AS") {
            Some(self.identifier()?)
        } else if let Token::Ident(name) = self.peek().clone() {
            // A bare identifier straight after an expression is an alias.
            self.next();
            Some(name)
        } else {
            None
        };
        Ok(SelectItem::Expr { expr, alias })
    }

    fn parse_table_ref(&mut self) -> Result<TableRef> {
        let mut left = self.parse_table_factor()?;
        loop {
            let join_type = if self.eat_keyword("CROSS") {
                self.expect_keyword("JOIN")?;
                JoinType::Cross
            } else if self.eat_keyword("INNER") {
                self.expect_keyword("JOIN")?;
                JoinType::Inner
            } else if self.eat_keyword("LEFT") {
                self.eat_keyword("OUTER");
                self.expect_keyword("JOIN")?;
                JoinType::Left
            } else if self.eat_keyword("RIGHT") {
                self.eat_keyword("OUTER");
                self.expect_keyword("JOIN")?;
                JoinType::Right
            } else if self.eat_keyword("FULL") {
                self.eat_keyword("OUTER");
                self.expect_keyword("JOIN")?;
                JoinType::Full
            } else if self.eat_keyword("JOIN") {
                JoinType::Inner
            } else if self.eat(&Token::Comma) {
                // `FROM a, b` is a cross join; the WHERE clause usually turns
                // it back into an inner join during optimisation.
                JoinType::Cross
            } else {
                return Ok(left);
            };

            let right = self.parse_table_factor()?;
            let on = if self.eat_keyword("ON") {
                Some(self.parse_expr(0)?)
            } else {
                None
            };
            if on.is_none() && join_type != JoinType::Cross {
                return Err(Error::parse(format!("{join_type} JOIN needs an ON clause")));
            }
            left = TableRef::Join {
                left: Box::new(left),
                right: Box::new(right),
                join_type,
                on,
            };
        }
    }

    fn parse_table_factor(&mut self) -> Result<TableRef> {
        if self.eat(&Token::LParen) {
            let inner = self.parse_table_ref()?;
            self.expect(&Token::RParen)?;
            return Ok(inner);
        }
        let name = self.identifier()?;
        let alias = if self.eat_keyword("AS") {
            Some(self.identifier()?)
        } else if let Token::Ident(a) = self.peek().clone() {
            self.next();
            Some(a)
        } else {
            None
        };
        Ok(TableRef::Table { name, alias })
    }

    // ---- expressions ----

    fn parse_expr(&mut self, min_prec: u8) -> Result<Expr> {
        let mut left = self.parse_prefix()?;
        loop {
            // Postfix forms bind at comparison precedence.
            if PREC_COMPARE >= min_prec {
                if let Some(e) = self.try_parse_postfix(&left)? {
                    left = e;
                    continue;
                }
            }
            let Some((op, prec)) = self.peek_binary_op() else {
                return Ok(left);
            };
            if prec < min_prec {
                return Ok(left);
            }
            self.next();
            // Left-associative: the right operand must bind more tightly.
            let right = self.parse_expr(prec + 1)?;
            left = Expr::binary(left, op, right);
        }
    }

    fn peek_binary_op(&self) -> Option<(BinaryOp, u8)> {
        Some(match self.peek() {
            Token::Keyword(k) if k == "OR" => (BinaryOp::Or, PREC_OR),
            Token::Keyword(k) if k == "AND" => (BinaryOp::And, PREC_AND),
            Token::Eq => (BinaryOp::Eq, PREC_COMPARE),
            Token::NotEq => (BinaryOp::NotEq, PREC_COMPARE),
            Token::Lt => (BinaryOp::Lt, PREC_COMPARE),
            Token::LtEq => (BinaryOp::LtEq, PREC_COMPARE),
            Token::Gt => (BinaryOp::Gt, PREC_COMPARE),
            Token::GtEq => (BinaryOp::GtEq, PREC_COMPARE),
            Token::Plus => (BinaryOp::Plus, PREC_ADD),
            Token::Minus => (BinaryOp::Minus, PREC_ADD),
            Token::Star => (BinaryOp::Multiply, PREC_MUL),
            Token::Slash => (BinaryOp::Divide, PREC_MUL),
            Token::Percent => (BinaryOp::Modulo, PREC_MUL),
            _ => return None,
        })
    }

    /// `IS [NOT] NULL`, `[NOT] BETWEEN`, `[NOT] IN`, `[NOT] LIKE`.
    fn try_parse_postfix(&mut self, left: &Expr) -> Result<Option<Expr>> {
        if self.eat_keyword("IS") {
            let negated = self.eat_keyword("NOT");
            self.expect_keyword("NULL")?;
            return Ok(Some(Expr::IsNull {
                expr: Box::new(left.clone()),
                negated,
            }));
        }

        // A `NOT` here belongs to the postfix form that follows it.
        let negated = if self.peek_keyword("NOT")
            && matches!(self.peek_at(1), Token::Keyword(k) if k == "BETWEEN" || k == "IN" || k == "LIKE")
        {
            self.next();
            true
        } else {
            false
        };

        if self.eat_keyword("BETWEEN") {
            // BETWEEN's bounds must not swallow the AND that separates them.
            let low = self.parse_expr(PREC_COMPARE)?;
            self.expect_keyword("AND")?;
            let high = self.parse_expr(PREC_COMPARE)?;
            return Ok(Some(Expr::Between {
                expr: Box::new(left.clone()),
                low: Box::new(low),
                high: Box::new(high),
                negated,
            }));
        }

        if self.eat_keyword("IN") {
            self.expect(&Token::LParen)?;
            let mut list = Vec::new();
            loop {
                list.push(self.parse_expr(0)?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RParen)?;
            return Ok(Some(Expr::InList {
                expr: Box::new(left.clone()),
                list,
                negated,
            }));
        }

        if self.eat_keyword("LIKE") {
            let pattern = self.parse_expr(PREC_COMPARE + 1)?;
            return Ok(Some(Expr::Like {
                expr: Box::new(left.clone()),
                pattern: Box::new(pattern),
                negated,
            }));
        }

        if negated {
            return Err(self.unexpected("BETWEEN, IN or LIKE after NOT"));
        }
        Ok(None)
    }

    fn parse_prefix(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Token::Minus => {
                self.next();
                Ok(Expr::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(self.parse_expr(PREC_MUL + 1)?),
                })
            }
            Token::Plus => {
                self.next();
                self.parse_expr(PREC_MUL + 1)
            }
            Token::Keyword(k) if k == "NOT" => {
                self.next();
                Ok(Expr::Unary {
                    op: UnaryOp::Not,
                    expr: Box::new(self.parse_expr(PREC_NOT)?),
                })
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        match self.peek().clone() {
            Token::Int(i) => {
                self.next();
                Ok(Expr::Literal(Value::Int64(i)))
            }
            Token::Float(x) => {
                self.next();
                Ok(Expr::Literal(Value::Float64(x)))
            }
            Token::Str(s) => {
                self.next();
                Ok(Expr::Literal(Value::Utf8(s)))
            }
            Token::Star => {
                self.next();
                Ok(Expr::Wildcard)
            }
            Token::LParen => {
                self.next();
                let e = self.parse_expr(0)?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }
            Token::Keyword(k) => match k.as_str() {
                "TRUE" => {
                    self.next();
                    Ok(Expr::Literal(Value::Boolean(true)))
                }
                "FALSE" => {
                    self.next();
                    Ok(Expr::Literal(Value::Boolean(false)))
                }
                "NULL" => {
                    self.next();
                    Ok(Expr::Literal(Value::Null))
                }
                "CAST" => self.parse_cast(),
                "CASE" => self.parse_case(),
                _ => Err(self.unexpected("an expression")),
            },
            Token::Ident(name) => {
                self.next();
                if self.eat(&Token::LParen) {
                    return self.parse_function_args(name);
                }
                if self.eat(&Token::Dot) {
                    let col = self.identifier()?;
                    return Ok(Expr::qualified(name, col));
                }
                Ok(Expr::column(name))
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    fn parse_function_args(&mut self, name: String) -> Result<Expr> {
        let distinct = self.eat_keyword("DISTINCT");
        let mut args = Vec::new();
        if !self.eat(&Token::RParen) {
            loop {
                args.push(self.parse_expr(0)?);
                if !self.eat(&Token::Comma) {
                    break;
                }
            }
            self.expect(&Token::RParen)?;
        }
        if distinct && args.len() != 1 {
            return Err(Error::parse(format!(
                "{name}(DISTINCT ...) takes exactly one argument"
            )));
        }
        Ok(Expr::Function {
            name,
            args,
            distinct,
        })
    }

    fn parse_cast(&mut self) -> Result<Expr> {
        self.expect_keyword("CAST")?;
        self.expect(&Token::LParen)?;
        let expr = self.parse_expr(0)?;
        self.expect_keyword("AS")?;
        let type_name = match self.next() {
            Token::Ident(t) => t,
            other => {
                return Err(Error::parse(format!(
                    "expected a type name in CAST, found `{other}`"
                )))
            }
        };
        let data_type = DataType::from_sql_name(&type_name)?;
        self.expect(&Token::RParen)?;
        Ok(Expr::Cast {
            expr: Box::new(expr),
            data_type,
        })
    }

    fn parse_case(&mut self) -> Result<Expr> {
        self.expect_keyword("CASE")?;
        let operand = if self.peek_keyword("WHEN") {
            None
        } else {
            Some(Box::new(self.parse_expr(0)?))
        };
        let mut branches = Vec::new();
        while self.eat_keyword("WHEN") {
            let when = self.parse_expr(0)?;
            self.expect_keyword("THEN")?;
            let then = self.parse_expr(0)?;
            branches.push((when, then));
        }
        if branches.is_empty() {
            return Err(Error::parse("CASE needs at least one WHEN branch"));
        }
        let else_result = if self.eat_keyword("ELSE") {
            Some(Box::new(self.parse_expr(0)?))
        } else {
            None
        };
        self.expect_keyword("END")?;
        Ok(Expr::Case {
            operand,
            branches,
            else_result,
        })
    }
}

/// Parses one statement from `sql`.
pub fn parse(sql: &str) -> Result<Statement> {
    Parser::parse_one(sql)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(sql: &str) -> Query {
        match parse(sql).unwrap() {
            Statement::Query(q) => *q,
            other => panic!("expected a query, got {other:?}"),
        }
    }

    fn expr(sql: &str) -> String {
        let q = query(&format!("SELECT {sql}"));
        match &q.projection[0] {
            SelectItem::Expr { expr, .. } => expr.to_string(),
            other => panic!("expected an expression, got {other:?}"),
        }
    }

    #[test]
    fn a_minimal_select_parses() {
        let q = query("SELECT a, b FROM t");
        assert_eq!(q.projection.len(), 2);
        assert!(matches!(
            q.from,
            Some(TableRef::Table { ref name, .. }) if name == "t"
        ));
        assert!(q.selection.is_none());
        assert!(!q.distinct);
    }

    #[test]
    fn multiplication_binds_tighter_than_addition() {
        assert_eq!(expr("1 + 2 * 3"), "(1 + (2 * 3))");
        assert_eq!(expr("(1 + 2) * 3"), "((1 + 2) * 3)");
    }

    #[test]
    fn arithmetic_is_left_associative() {
        assert_eq!(expr("1 - 2 - 3"), "((1 - 2) - 3)");
        assert_eq!(expr("8 / 4 / 2"), "((8 / 4) / 2)");
    }

    #[test]
    fn and_binds_tighter_than_or() {
        assert_eq!(expr("a OR b AND c"), "(a OR (b AND c))");
        assert_eq!(expr("(a OR b) AND c"), "((a OR b) AND c)");
    }

    #[test]
    fn comparison_binds_tighter_than_and() {
        assert_eq!(expr("a = 1 AND b = 2"), "((a = 1) AND (b = 2))");
    }

    #[test]
    fn not_binds_looser_than_comparison_and_tighter_than_and() {
        assert_eq!(expr("NOT a = 1"), "NOT (a = 1)");
        assert_eq!(expr("NOT a AND b"), "(NOT a AND b)");
    }

    #[test]
    fn unary_minus_binds_tighter_than_multiplication() {
        assert_eq!(expr("-a * b"), "(-a * b)");
        assert_eq!(expr("- 2 + 3"), "(-2 + 3)");
        assert_eq!(expr("+5"), "5");
    }

    #[test]
    fn between_does_not_swallow_the_and_that_separates_its_bounds() {
        assert_eq!(expr("a BETWEEN 1 AND 5"), "a BETWEEN 1 AND 5");
        // The trailing AND belongs to the outer conjunction, not to BETWEEN.
        assert_eq!(
            expr("a BETWEEN 1 AND 5 AND b = 2"),
            "(a BETWEEN 1 AND 5 AND (b = 2))"
        );
        assert_eq!(expr("a NOT BETWEEN 1 AND 5"), "a NOT BETWEEN 1 AND 5");
    }

    #[test]
    fn the_postfix_forms_parse_with_and_without_not() {
        assert_eq!(expr("a IS NULL"), "a IS NULL");
        assert_eq!(expr("a IS NOT NULL"), "a IS NOT NULL");
        assert_eq!(expr("a IN (1, 2)"), "a IN (1, 2)");
        assert_eq!(expr("a NOT IN (1)"), "a NOT IN (1)");
        assert_eq!(expr("a LIKE 'x%'"), "a LIKE 'x%'");
        assert_eq!(expr("a NOT LIKE 'x%'"), "a NOT LIKE 'x%'");
    }

    #[test]
    fn postfix_forms_chain_with_comparisons() {
        assert_eq!(expr("a IS NULL AND b IN (1)"), "(a IS NULL AND b IN (1))");
    }

    #[test]
    fn a_dangling_not_before_something_else_is_reported() {
        assert!(parse("SELECT a NOT b FROM t").is_err());
    }

    #[test]
    fn functions_parse_with_wildcards_distinct_and_nesting() {
        assert_eq!(expr("count(*)"), "count(*)");
        assert_eq!(expr("count(DISTINCT a)"), "count(DISTINCT a)");
        assert_eq!(expr("sum(a + b)"), "sum((a + b))");
        assert_eq!(expr("coalesce()"), "coalesce()");
    }

    #[test]
    fn distinct_with_several_arguments_is_refused() {
        assert!(parse("SELECT count(DISTINCT a, b) FROM t").is_err());
    }

    #[test]
    fn cast_and_case_parse() {
        assert_eq!(expr("CAST(a AS INT)"), "CAST(a AS INT64)");
        assert_eq!(
            expr("CASE WHEN a > 1 THEN 'big' ELSE 'small' END"),
            "CASE WHEN (a > 1) THEN 'big' ELSE 'small' END"
        );
        assert_eq!(
            expr("CASE a WHEN 1 THEN 'one' END"),
            "CASE a WHEN 1 THEN 'one' END"
        );
    }

    #[test]
    fn a_case_without_branches_or_a_bad_cast_type_is_refused() {
        assert!(parse("SELECT CASE END FROM t").is_err());
        assert!(parse("SELECT CAST(a AS blob) FROM t").is_err());
        assert!(parse("SELECT CAST(a AS 'INT') FROM t").is_err());
    }

    #[test]
    fn qualified_columns_and_wildcards_parse() {
        let q = query("SELECT t.*, o.id, * FROM t");
        assert!(matches!(q.projection[0], SelectItem::QualifiedWildcard(ref n) if n == "t"));
        assert!(matches!(q.projection[2], SelectItem::Wildcard));
        match &q.projection[1] {
            SelectItem::Expr { expr, .. } => assert_eq!(expr.to_string(), "o.id"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn aliases_parse_with_and_without_as() {
        let q = query("SELECT a AS x, b y FROM t");
        match (&q.projection[0], &q.projection[1]) {
            (SelectItem::Expr { alias: Some(a), .. }, SelectItem::Expr { alias: Some(b), .. }) => {
                assert_eq!(a, "x");
                assert_eq!(b, "y");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn every_join_flavour_parses() {
        for (sql, want) in [
            ("SELECT * FROM a JOIN b ON a.i = b.i", JoinType::Inner),
            ("SELECT * FROM a INNER JOIN b ON a.i = b.i", JoinType::Inner),
            ("SELECT * FROM a LEFT JOIN b ON a.i = b.i", JoinType::Left),
            (
                "SELECT * FROM a LEFT OUTER JOIN b ON a.i = b.i",
                JoinType::Left,
            ),
            ("SELECT * FROM a RIGHT JOIN b ON a.i = b.i", JoinType::Right),
            ("SELECT * FROM a FULL JOIN b ON a.i = b.i", JoinType::Full),
            ("SELECT * FROM a CROSS JOIN b", JoinType::Cross),
            ("SELECT * FROM a, b", JoinType::Cross),
        ] {
            match query(sql).from.unwrap() {
                TableRef::Join { join_type, .. } => assert_eq!(join_type, want, "{sql}"),
                other => panic!("{sql} produced {other:?}"),
            }
        }
    }

    #[test]
    fn an_inner_join_without_on_is_refused() {
        assert!(parse("SELECT * FROM a JOIN b")
            .unwrap_err()
            .to_string()
            .contains("needs an ON clause"));
    }

    #[test]
    fn joins_are_left_deep_and_table_aliases_parse() {
        let q = query("SELECT * FROM a x JOIN b AS y ON x.i = y.i JOIN c ON y.i = c.i");
        match q.from.unwrap() {
            TableRef::Join { left, .. } => {
                assert!(matches!(*left, TableRef::Join { .. }));
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parenthesised_from_clauses_parse() {
        let q = query("SELECT * FROM (a JOIN b ON a.i = b.i)");
        assert!(matches!(q.from, Some(TableRef::Join { .. })));
    }

    #[test]
    fn the_full_clause_list_parses_in_order() {
        let q = query(
            "SELECT region, count(*) c FROM sales WHERE amount > 10 \
             GROUP BY region HAVING count(*) > 2 ORDER BY c DESC LIMIT 5 OFFSET 10",
        );
        assert_eq!(q.group_by.len(), 1);
        assert!(q.having.is_some());
        assert_eq!(q.order_by.len(), 1);
        assert!(!q.order_by[0].ascending);
        assert_eq!(q.limit, Some(5));
        assert_eq!(q.offset, Some(10));
        assert!(q.selection.is_some());
    }

    #[test]
    fn order_by_defaults_nulls_last_ascending_and_first_descending() {
        assert!(!query("SELECT a FROM t ORDER BY a").order_by[0].nulls_first);
        assert!(query("SELECT a FROM t ORDER BY a DESC").order_by[0].nulls_first);
        assert!(query("SELECT a FROM t ORDER BY a NULLS FIRST").order_by[0].nulls_first);
        assert!(!query("SELECT a FROM t ORDER BY a DESC NULLS LAST").order_by[0].nulls_first);
    }

    #[test]
    fn a_having_that_should_have_been_a_where_is_refused() {
        assert!(parse("SELECT a FROM t HAVING a > 1")
            .unwrap_err()
            .to_string()
            .contains("did you mean WHERE"));
        // ...but an aggregate makes a bare HAVING meaningful.
        assert!(parse("SELECT count(*) FROM t HAVING count(*) > 1").is_ok());
    }

    #[test]
    fn a_negative_limit_is_refused() {
        assert!(parse("SELECT a FROM t LIMIT -1").is_err());
        assert!(parse("SELECT a FROM t LIMIT 'x'").is_err());
    }

    #[test]
    fn select_without_from_parses() {
        let q = query("SELECT 1 + 1");
        assert!(q.from.is_none());
    }

    #[test]
    fn create_table_parses_types_and_nullability() {
        match parse("CREATE TABLE t (id INT NOT NULL, name TEXT, score DOUBLE)").unwrap() {
            Statement::CreateTable { name, columns } => {
                assert_eq!(name, "t");
                assert_eq!(columns.len(), 3);
                assert!(!columns[0].nullable);
                assert!(columns[1].nullable);
                assert_eq!(columns[2].data_type, DataType::Float64);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn insert_copy_and_drop_parse() {
        match parse("INSERT INTO t VALUES (1, 'a'), (2, 'b')").unwrap() {
            Statement::Insert { table, rows } => {
                assert_eq!(table, "t");
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse("COPY sales FROM 'sales.csv'").unwrap() {
            Statement::Copy { table, path } => {
                assert_eq!(table, "sales");
                assert_eq!(path, "sales.csv");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            parse("DROP TABLE t").unwrap(),
            Statement::DropTable { .. }
        ));
    }

    #[test]
    fn copy_without_a_quoted_path_is_refused() {
        assert!(parse("COPY t FROM sales.csv")
            .unwrap_err()
            .to_string()
            .contains("quoted file path"));
    }

    #[test]
    fn explain_and_explain_analyze_parse() {
        match parse("EXPLAIN SELECT a FROM t").unwrap() {
            Statement::Explain { analyze, .. } => assert!(!analyze),
            other => panic!("unexpected {other:?}"),
        }
        match parse("EXPLAIN ANALYZE SELECT a FROM t").unwrap() {
            Statement::Explain { analyze, .. } => assert!(analyze),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn several_statements_parse_and_empty_ones_are_skipped() {
        let stmts = Parser::parse_statements("SELECT 1; ; SELECT 2;;").unwrap();
        assert_eq!(stmts.len(), 2);
        assert!(Parser::parse_statements("  ").unwrap().is_empty());
        assert!(Parser::parse_one("").is_err());
        assert!(Parser::parse_one("SELECT 1; SELECT 2").is_err());
    }

    #[test]
    fn errors_name_what_was_expected_and_where() {
        let err = parse("SELECT FROM").unwrap_err();
        assert!(err.to_string().contains("expected an expression"));
        assert!(err.to_string().contains("offset"));

        assert!(parse("SELECT a FROM").is_err());
        assert!(parse("SELECT a FROM t WHERE").is_err());
        assert!(parse("SELECT (a FROM t").is_err());
        assert!(parse("UPDATE t SET a = 1").is_err());
        assert!(parse("SELECT a FROM t GROUP a").is_err());
        assert!(parse("SELECT a FROM t ORDER a").is_err());
        assert!(parse("CREATE TABLE t").is_err());
        assert!(parse("INSERT INTO t (1)").is_err());
    }

    #[test]
    fn a_statement_followed_by_garbage_is_refused() {
        assert!(parse("SELECT a FROM t garbage extra").is_err());
    }

    #[test]
    fn quoted_identifiers_survive_into_the_tree() {
        let q = query(r#"SELECT "odd name" FROM "my table""#);
        match &q.projection[0] {
            SelectItem::Expr { expr, .. } => assert_eq!(expr.to_string(), "odd name"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(matches!(
            q.from,
            Some(TableRef::Table { ref name, .. }) if name == "my table"
        ));
    }
}
