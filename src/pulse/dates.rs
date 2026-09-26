//! Calendar dates for git timestamps (UTC), without a date crate.

/// Convert epoch seconds to YYYY-MM-DD string
pub fn epoch_to_date_string(epoch_secs: i64) -> String {
    // Simple date calculation without external deps
    let days = epoch_secs / 86400;
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert days since epoch to (year, month, day)
pub fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_to_date_string() {
        // 2024-01-01 00:00:00 UTC = 1704067200
        assert_eq!(epoch_to_date_string(1704067200), "2024-01-01");
    }

    #[test]
    fn test_days_to_ymd() {
        let (y, m, d) = days_to_ymd(0); // 1970-01-01
        assert_eq!((y, m, d), (1970, 1, 1));
    }
}
