# Manager Sokovan layer — Guardrails

> Sokovan is the coordinator layer that advances the deployment / route lifecycle in ticks.
> For the operating model and rationale, see `CONTEXTS.md` in the same directory (and the `README.md` below it).

## Handler file layout

- **one handler = one file**: every `*Handler` class under `sokovan/**/handlers/` lives in its own module.
- The file name follows the lifecycle stage/sub-step that the handler deals with
  (example: `deploying_provisioning.py`, `deploying_rolling_back.py`, `warming_up.py`, `terminating.py`).
- `base.py` is reserved for the abstract `*Handler` base class. Put shared utilities in a separate module — do NOT put them in `base.py`.
- Do NOT put implementations in `__init__.py` (re-exports follow the root global rules).

## Handler vs coordinator responsibility

- The handler judges and returns only the result of its action (success / failure / skip, etc.). Do NOT put
  history-based judgment such as retry / timeout / give_up in the handler — the coordinator handles them centrally.
- Return the failure reason wrapped in `DeploymentExecutionError`.

## Status transitions

- `status_transitions()` must declare the target lifecycle for each result the coordinator classifies
  (`success`, `need_retry`, `expired`, `give_up`). A result with no target stays in the current status (`coordinator._handle_status_transitions`).
- A transition moves only rows still in the status they were read in. Guard every status write with
  the status the handler read (`SessionTransitionInfo.from_status`), and drive history, broadcasts,
  hooks and follow-up work from the rows that actually moved, never from the rows requested. A
  handler that takes no lock (for example the stuck-start expiry) relies on this guard.
