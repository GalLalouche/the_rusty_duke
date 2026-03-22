//! Shared CLI argument parsing helpers for training binaries.

/// Parse a typed flag value from CLI args: `--flag <value>`.
///
/// Returns `None` if the flag is missing or the value fails to parse.
pub fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}
