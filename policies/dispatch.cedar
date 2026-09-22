// Only broker-computed scope and concrete approvals can permit dispatch.
permit(principal, action == Action::"dispatch", resource)
when { context.exactApproval || (context.scoped && !context.requiresExact) };

forbid(principal, action, resource)
when { context.prohibited || context.expired };
