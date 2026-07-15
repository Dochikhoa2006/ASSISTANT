"""Exact keyword policy for deterministic knowledge and reminder actions."""

from __future__ import annotations


DELETE_ACTION_KEYWORDS: tuple[str, ...] = (
    "delete", "remove", "erase", "clear", "discard", "drop", "destroy",
    "eliminate", "purge", "wipe", "wipe out", "get rid of", "take out",
    "remove permanently", "delete permanently", "forget", "forget this",
    "forget that", "forget about", "remove from memory", "erase from memory",
    "clear from memory", "delete from memory", "remove the record",
    "delete the record", "remove the entry", "delete the entry", "remove the item",
    "delete the item", "remove the reminder", "delete the reminder",
    "cancel and delete", "archive and remove", "mark as deleted", "soft delete",
    "hard delete",
)


MODIFY_ACTION_KEYWORDS: tuple[str, ...] = (
    "replace", "modify", "change", "update", "edit", "revise", "rewrite",
    "correct", "fix", "adjust", "alter", "amend", "refine", "rework",
    "rephrase", "rename", "reschedule", "move", "shift", "postpone",
    "bring forward", "change to", "replace with", "swap with", "switch to",
    "set to", "update to", "edit to", "correct to", "revise to",
    "change the value", "change the date", "change the time",
    "change the subject", "change the content", "change the reminder",
    "modify the reminder", "update the reminder", "edit the reminder",
    "replace the old one", "replace the existing one", "overwrite",
    "overwrite with", "supersede", "substitute", "patch", "refresh",
)


ADD_ACTION_KEYWORDS: tuple[str, ...] = (
    "add", "create", "insert", "save", "store", "remember", "remember that", "keep",
    "retain", "register", "include", "append", "attach", "enter", "log",
    "capture", "note that", "write down",
    "put in", "put into", "add to", "save to", "store in", "record in",
    "record that",
    "create a new", "add a new", "insert a new", "set a reminder",
    "create a reminder", "add a reminder", "schedule a reminder", "schedule",
    "remind me", "remember this", "save this", "store this", "record this",
    "add this", "keep this in memory", "add to memory", "save to memory",
    "store as knowledge", "ingest", "import",
)


TURN_ON_ACTION_KEYWORDS: tuple[str, ...] = (
    "turn on", "switch on", "enable", "activate", "reactivate", "resume",
    "restart", "start", "restore", "re-enable", "re-enable it", "enable again",
    "activate again", "turn back on", "switch back on", "start again",
    "resume it", "resume the reminder", "activate the reminder",
    "enable the reminder", "turn on the reminder", "set to active",
    "mark as active", "mark as enabled", "set as enabled", "set as active",
    "unsuspend", "unpause", "continue", "continue the reminder",
    "restore the reminder", "reinstate", "reinstate the reminder",
)


TURN_OFF_ACTION_KEYWORDS: tuple[str, ...] = (
    "turn off", "switch off", "disable", "deactivate", "pause", "stop",
    "suspend", "mute", "silence", "cancel", "halt", "end", "turn it off",
    "switch it off", "disable it", "deactivate it", "pause it", "stop it",
    "turn off the reminder", "disable the reminder", "deactivate the reminder",
    "pause the reminder", "stop the reminder", "cancel the reminder",
    "set to inactive", "mark as inactive", "mark as disabled", "set as disabled",
    "snooze indefinitely", "temporarily disable", "temporarily turn off",
    "suspend the reminder", "do not remind me", "stop reminding me",
)


ACTION_KEYWORDS: tuple[str, ...] = (
    *DELETE_ACTION_KEYWORDS,
    *MODIFY_ACTION_KEYWORDS,
    *ADD_ACTION_KEYWORDS,
    *TURN_ON_ACTION_KEYWORDS,
    *TURN_OFF_ACTION_KEYWORDS,
)
