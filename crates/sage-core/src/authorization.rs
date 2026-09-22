//! Cedar is an additional mandatory dispatch gate. Its inputs are computed by
//! the broker; neither a model nor a plugin supplies policy source or facts.
use crate::{CoreError, CoreResult};
use cedar_policy::{
    Authorizer, Context, Decision, Entities, PolicySet, Request, Schema, ValidationMode, Validator,
};
use std::sync::OnceLock;

const POLICY: &str = include_str!("../../../policies/dispatch.cedar");
struct BasePolicy {
    policies: PolicySet,
    schema: Schema,
}
static BASE: OnceLock<Result<BasePolicy, String>> = OnceLock::new();

fn base() -> CoreResult<&'static BasePolicy> {
    BASE.get_or_init(|| {
        let policies: PolicySet = POLICY.parse().map_err(|e| format!("{e}"))?;
        let schema = Schema::from_json_value(serde_json::json!({"": {
            "entityTypes": {"User":{}, "Resource":{}},
            "actions": {"dispatch": {"appliesTo": {
                "principalTypes":["User"], "resourceTypes":["Resource"],
                "context":{"type":"Record","attributes": {
                    "scoped":{"type":"Boolean"}, "exactApproval":{"type":"Boolean"},
                    "requiresExact":{"type":"Boolean"}, "prohibited":{"type":"Boolean"},
                    "expired":{"type":"Boolean"}
                }}
            }}}
        }}))
        .map_err(|e| e.to_string())?;
        if !Validator::new(schema.clone())
            .validate(&policies, ValidationMode::Strict)
            .validation_passed()
        {
            return Err("Bundled authorization policy failed schema validation".into());
        }
        Ok(BasePolicy { policies, schema })
    })
    .as_ref()
    .map_err(|e| CoreError::PolicyDenied(e.clone()))
}

pub fn authorize(
    scoped: bool,
    exact_approval: bool,
    requires_exact: bool,
    prohibited: bool,
    expired: bool,
) -> CoreResult<()> {
    let base = base()?;
    let context = Context::from_json_value(
        serde_json::json!({
            "scoped":scoped,"exactApproval":exact_approval,"requiresExact":requires_exact,
            "prohibited":prohibited,"expired":expired
        }),
        None,
    )
    .map_err(|_| CoreError::PolicyDenied("Invalid authorization context".into()))?;
    let request = Request::new(
        "User::\"local-user\"".parse().unwrap(),
        "Action::\"dispatch\"".parse().unwrap(),
        "Resource::\"prepared-action\"".parse().unwrap(),
        context,
        Some(&base.schema),
    )
    .map_err(|_| CoreError::PolicyDenied("Invalid authorization request".into()))?;
    if Authorizer::new()
        .is_authorized(&request, &base.policies, &Entities::empty())
        .decision()
        != Decision::Allow
    {
        return Err(CoreError::PolicyDenied(
            "Dispatch is outside the approved scope".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_policy_input_combination_bypasses_denial() {
        for bits in 0..32 {
            let s = bits & 1 != 0;
            let a = bits & 2 != 0;
            let r = bits & 4 != 0;
            let p = bits & 8 != 0;
            let e = bits & 16 != 0;
            assert_eq!(
                authorize(s, a, r, p, e).is_ok(),
                !p && !e && (a || (s && !r)),
                "{bits}"
            );
        }
    }
}
