# Provider and tool development

## Model providers

Implement ModelProvider in the Rust core or in a separately supervised adapter that the core wraps.

The minimum contract is:

~~~text
descriptor() -> ProviderDescriptor
create_plan(PlanningContext) -> ActionGraph
replan(ReplanContext) -> ActionGraph
~~~

ASR, TTS, and vision use separate traits. Do not add optional speech fields to the reasoning call merely because one vendor supports them.

## Action additions

Adding an Action variant requires all of the following:

1. define typed arguments in domain/action.rs;
2. choose its ExecutionDomain;
3. define reversible behavior honestly;
4. add a redacted summary;
5. classify risk in policy.rs;
6. add prohibited cases where applicable;
7. define exact capability requirements;
8. canonicalize any resource paths;
9. add compiler candidates in priority order;
10. implement one restricted executor;
11. implement independent observation;
12. implement verifier matching;
13. add protocol/UI events if user interaction is needed;
14. add end-to-end tests.

A schema name alone is not an implemented tool.

## Tool registry

ToolDescriptor contains:

- name/version;
- input/output schema;
- risk;
- required capabilities;
- platforms;
- confirmation requirement;
- executor;
- timeout;
- verification strategy.

Registration never grants trust. The policy engine still evaluates every action instance.

## Browser integrations

Browser integrations belong in sage-browser-worker. They must:

- use a user-authorized session;
- bind exact origin;
- prefer DOM/structured APIs;
- use stable semantic element IDs;
- reject stale targets;
- tag page content untrusted;
- restrict file uploads to the granted path;
- verify URL/form/download state independently;
- never return cookies, passwords, or protected field values to the model.

## Sandbox tools

Sandbox additions must declare:

- exact executable;
- argument schema;
- allowed mounts;
- network destinations or no network;
- environment allowlist;
- CPU/memory/process/time/output ceilings;
- verification outputs.

Do not add a shell script string field.

## Privileged operations

Each operation needs a named request structure and independent validation in the helper. The helper must reject unknown fields/operations, authenticate its caller, and remain much smaller than sage-core.

Privilege is not an executor fallback. If an operation can run safely as the user, it does not belong in the helper.

## Tests

At minimum test:

- malformed model output;
- provenance loss;
- dependency cycles;
- policy category;
- approval mismatch/replay;
- expired/revoked capability;
- cross-task capability use;
- symlink/path escape;
- executor/resource mismatch;
- observation failure;
- verifier false positive;
- bounded replan;
- rollback conflict;
- frame size/version/authentication.

Platform-native code also needs a physical OS acceptance run. Rust tests on macOS do not prove Windows behavior.
