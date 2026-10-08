//! Sage's mandatory dispatch gate over broker-computed scope and authority.
use crate::{CoreError, CoreResult};

/// Enforce the final, broker-owned scope gate before a prepared action can be
/// dispatched. A concrete exact approval may authorize an exact-only action;
/// ordinary scope is sufficient only when the action does not require exact
/// approval. Prohibition and expiry override every allow condition.
pub fn authorize(
    scoped: bool,
    exact_approval: bool,
    requires_exact: bool,
    prohibited: bool,
    expired: bool,
) -> CoreResult<()> {
    if !prohibited && !expired && (exact_approval || (scoped && !requires_exact)) {
        Ok(())
    } else {
        Err(CoreError::PolicyDenied(
            "Dispatch is outside the approved scope".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_policy_input_combination_bypasses_denial() {
        for bits in 0..32 {
            let scoped = bits & 1 != 0;
            let exact_approval = bits & 2 != 0;
            let requires_exact = bits & 4 != 0;
            let prohibited = bits & 8 != 0;
            let expired = bits & 16 != 0;
            let expected =
                !prohibited && !expired && (exact_approval || (scoped && !requires_exact));

            assert_eq!(
                authorize(scoped, exact_approval, requires_exact, prohibited, expired).is_ok(),
                expected,
                "authorization input bitset {bits}"
            );
        }
    }

    #[test]
    fn exact_approval_cannot_override_prohibition_or_expiry() {
        assert!(authorize(true, true, true, true, false).is_err());
        assert!(authorize(true, true, true, false, true).is_err());
    }
}
