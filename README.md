# queryforge

[![CI](https://github.com/sahilkalgutkar/queryforge/actions/workflows/ci.yml/badge.svg)](https://github.com/sahilkalgutkar/queryforge/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sahilkalgutkar/queryforge/branch/main/graph/badge.svg)](https://codecov.io/gh/sahilkalgutkar/queryforge)
[![patch coverage](https://img.shields.io/badge/patch%20coverage-min%2080%25-blue.svg)](codecov.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust 1.82+](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org/)

I built queryforge to understand what actually happens between typing a SQL
query and getting rows back — by writing every stage of it in Rust: the lexer,
the parser, the binder, a cost-based optimiser, a vectorised execution engine,
and the columnar file format underneath all of it.

This repository is in progress; the sections below fill in as each layer lands.
