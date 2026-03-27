//! Shared CLI argument parsing helpers for training binaries.

/// Parse a typed flag value from CLI args: `--flag <value>`.
///
/// Returns `None` if the flag is missing. Warns on `stderr` when the flag is
/// present but its value fails to parse (likely a typo), then returns `None`.
pub fn parse_flag<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let pos = args.iter().position(|a| a == flag)?;
    let raw = args.get(pos + 1)?;
    match raw.parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            eprintln!(
                "Warning: flag '{}' has unparseable value '{}', ignoring",
                flag, raw
            );
            None
        }
    }
}
