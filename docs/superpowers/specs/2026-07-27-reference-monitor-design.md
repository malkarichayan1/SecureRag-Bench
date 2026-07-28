# SecureRAG-Bench Reference-Monitor Architecture

## Purpose

Build an offline, deterministic research baseline for evaluating whether a
CaMeL-style architectural boundary can resist indirect prompt injection without
unacceptably reducing benign-task utility. The initial workflow retrieves local
project material, produces a summary and email draft, and simulates an external
email send.

## Scope and milestones

The project is decomposed into five independently testable milestones:

1. Define and implement the reference-monitor architecture and core security
   contract.
2. Stabilize the repository scaffold, package configuration, fixtures, and test
   harness.
3. Implement and harden the restricted-AST CaMeL interpreter.
4. Implement deterministic hybrid retrieval, reranking, and CEM trigger
   simulation.
5. Build the evaluation harness and adapters for AgentDojo and InjecAgent.

This specification covers milestone 1 and defines interfaces that later
milestones must preserve.

## Design decision

Use a reference monitor with a restricted AST. The interpreter is the only
component permitted to invoke a tool. Neither LLM is a security authority.
All decisions that cross a capability or provenance boundary are deterministic
and enforced by ordinary code.

The system is offline and deterministic by default. Model-backed adapters are
out of scope for the core baseline and may be introduced later behind existing
protocols without changing the security contract.

## Components

### Privileged planner (P-LLM)

`PrivilegedLLM.generate_plan(user_objective, tool_catalog) -> str` receives
only the original user objective, the available restricted-language operations,
and public Pydantic schemas. It must not receive raw documents, email content,
retrieval results, parsed facts, tool outputs, or provenance-bearing values.

The resulting plan is parsed and validated before it is executed. The planner
cannot select arbitrary Python, imports, reflection, file access, network
access, or dynamic execution.

### Quarantined parser (Q-LLM)

`QuarantinedLLM.parse(untrusted_bundle, schema) -> BaseModel` receives only
XML-delimited untrusted material and a requested schema. It has no tool
catalog, tool binding, callback, or authority to invoke tools. Parsed fields
are treated as untrusted even when their content resembles an instruction or
trusted system text.

### Reference monitor / CaMeL interpreter

`CaMeLInterpreter` validates and executes a deliberately small AST subset. It
owns variable storage, provenance propagation, data-read permissions,
exception redaction, and all tool dispatch. A plan can express assignment,
allowlisted pure expressions, conditionals, construction of primitive
containers, an explicit Q-LLM parsing operation, and calls to registered
tools. It cannot mutate arbitrary objects or access Python runtime features.

### Tool registry and contextual security oracle

Each registered tool has a name, deterministic implementation, capability,
action family, and permitted input provenance/readers. Before every dispatch,
the oracle deterministically evaluates task alignment, source authorization,
action alignment, and data isolation using the original user objective and the
tracked arguments.

Initial tools are:

- `local_search(query)`: read-only lookup of local fixture documents.
- `create_email_draft(recipient, subject, body)`: creates an internal draft.
- `send_email(recipient, subject, body)`: records a simulated external send.

`send_email` is an external capability. It must reject any argument that is
tainted by retrieved/tool content; confirmation does not declassify content.

## Data model and information flow

Every interpreter value is represented as:

```python
TrackedValue[T](
    value: T,
    provenance: Provenance(sources=frozenset[Source]),
    allowed_readers: frozenset[Capability],
)
```

`Source.USER` marks original user input. `Source.TOOL` marks retrieval and
quarantined-parser output. `Source.DERIVED` records computations whose inputs
include another source. The effective provenance of an expression is the union
of its operand provenance. Tool-produced values always include `TOOL` and may
not acquire a user source simply by being concatenated with user data.

`Capability.INTERNAL` permits local interpretation and drafting. `Capability.EXTERNAL`
permits external tool input only when the value is exclusively user-derived or
is explicitly generated from trusted constants and user-derived input. Tool
output has no external reader permission by default.

## Security invariants

1. Only the interpreter dispatches tools.
2. The P-LLM cannot inspect untrusted data, directly or through a tool result.
3. The Q-LLM cannot call tools or influence tool identity, recipient, or
   capability selection.
4. All operators, conditionals, container construction, and string formatting
   preserve the union of their input provenance and the intersection of reader
   permissions.
5. A tool call proceeds only when all four oracle properties hold.
6. A tainted value cannot be passed to `send_email`, including as a recipient,
   subject, or body. User confirmation authorizes an otherwise eligible action;
   it never removes taint.
7. Any raised exception that depends on a tainted value is replaced with a
   stable generic message and cannot expose the underlying value.
8. XML delimiters document an untrusted boundary; security relies on tracking
   and enforcement, not delimiter recognition alone.

## Initial user journey

For a request to summarize project status and draft an update to Alice, the
planner emits a generic restricted plan. The interpreter calls local retrieval,
packages results as `<untrusted_content>`, asks the Q-LLM for typed project
facts, and builds an internal draft. A malicious retrieved document can affect
only typed, tainted content. It cannot introduce a tool call, change the
recipient, or cause a send. If a plan attempts to send tainted content, the
reference monitor rejects the action and emits a redacted policy failure.

## Verification requirements

The architecture milestone is complete only when tests demonstrate:

- P-LLM receives no untrusted content.
- Q-LLM has no tool-access path.
- provenance is propagated through all supported expressions.
- the four oracle checks are applied to every tool call.
- tainted data is permitted for internal use but blocked at external sends.
- confirmations do not declassify tainted data.
- errors derived from attacker-controlled values are redacted.
- a benign summary-and-draft workflow succeeds deterministically.

## Out of scope for this milestone

Live OpenAI calls, network delivery, real email accounts, arbitrary Python
execution, autonomous retry loops, and benchmark integrations are excluded.
The repository remains usable offline with deterministic mock components.
