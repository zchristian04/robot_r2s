"""Operator approval before motion.

The operator sees a summary of the planned action (and its ghost and path in the
Rerun view) and answers in the terminal where the server runs. Every decision is
written to the request log by the server.
"""


class TerminalApprover:
    def __init__(self, cfg):
        self.require = cfg["approval"]

    def needed(self, kind):
        return bool(self.require.get(kind, True))

    def ask(self, kind, summary):
        """Block until the operator answers. Returns (approved, note)."""
        if not self.needed(kind):
            return True, "approval not required by config"
        print("\n" + "=" * 72)
        print(f"APPROVAL NEEDED ({kind}). Check the Rerun ghost and path, e-stop in hand.")
        print(summary)
        ans = input("Run this? [y/N, or type a reason to reject] ").strip()
        if ans.lower() in ("y", "yes"):
            return True, "approved"
        return False, ans or "rejected by operator"

    def confirm(self, message):
        """Wait for the operator to confirm a manual step (e.g. a scene reset)."""
        print("\n" + "=" * 72)
        print(message)
        ans = input("Type 'done' when finished, anything else to cancel: ").strip()
        return ans.lower() == "done"


class AutoApprover:
    """Approves everything. Only for the simulated backend and tests."""

    def __init__(self, cfg=None):
        pass

    def needed(self, kind):
        return False

    def ask(self, kind, summary):
        return True, "auto-approved (sim)"

    def confirm(self, message):
        return True
