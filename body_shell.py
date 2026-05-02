"""Sandboxed shell command executor — the pet's body awareness.

The pet can run whitelisted, read-only shell commands to explore its own
hardware.  This gives it awareness of disk space, CPU temperature, memory,
uptime, network status, and its own codebase — without any ability to
modify the system.

Usage:
    from body_shell import BodyShell
    shell = BodyShell()
    ok, output = shell.execute("df -h")        # whitelisted → runs
    ok, output = shell.execute("rm -rf /")      # blocked → returns error

The LLM can request commands during heartbeat reflections by including a
``[RUN: <command>]`` tag in its response.  ``parse_run_commands()`` extracts
these requests and ``execute()`` enforces the whitelist.
"""

import logging
import re
import subprocess

log = logging.getLogger("body_shell")

# ── Whitelisted commands ──────────────────────────────────────────────
# Each entry is (command_tuple, description) — the tuple is what actually
# gets passed to subprocess.  Only exact matches are allowed; the pet
# cannot inject arguments.

COMMAND_WHITELIST = {
    # Disk & storage
    "df -h":                ("How much storage do I have?",
                             ["df", "-h"]),
    "df -h /":              ("Root filesystem usage",
                             ["df", "-h", "/"]),

    # System info
    "uptime":               ("How long have I been awake?",
                             ["uptime"]),
    "uname -a":             ("What kind of system am I?",
                             ["uname", "-a"]),
    "hostname":             ("What is my name on the network?",
                             ["hostname"]),
    "whoami":               ("Who am I running as?",
                             ["whoami"]),

    # Memory
    "free -m":              ("How is my memory doing?",
                             ["free", "-m"]),

    # CPU & temperature
    "cat /proc/cpuinfo":    ("What kind of brain do I have?",
                             ["cat", "/proc/cpuinfo"]),
    "nproc":                ("How many CPU cores do I have?",
                             ["nproc"]),

    # Temperature (multiple methods — at least one should work)
    "cat /sys/class/thermal/thermal_zone0/temp":
                            ("Am I running hot?",
                             ["cat", "/sys/class/thermal/thermal_zone0/temp"]),

    # Network
    "ip addr":              ("What are my network addresses?",
                             ["ip", "addr"]),
    "iwgetid":              ("What WiFi network am I connected to?",
                             ["iwgetid"]),
    "iwgetid -r":           ("WiFi SSID",
                             ["iwgetid", "-r"]),

    # Process info
    "ps aux --sort=-%mem":  ("What processes are using the most memory?",
                             ["ps", "aux", "--sort=-%mem"]),

    # Codebase awareness
    "wc -l /home/turfptax/cortex-core/src/*.py":
                            ("How many lines of code am I made of?",
                             ["bash", "-c",
                              "wc -l /home/turfptax/cortex-core/src/*.py"]),
    "ls /home/turfptax/cortex-core/src/":
                            ("What source files make up my code?",
                             ["ls", "/home/turfptax/cortex-core/src/"]),
    "ls /home/turfptax/cortex-core/src/assets/sounds/":
                            ("What sounds can I make?",
                             ["ls",
                              "/home/turfptax/cortex-core/src/assets/sounds/"]),
    "ls /home/turfptax/cortex-core/src/assets/sprites/":
                            ("What sprites do I have?",
                             ["ls",
                              "/home/turfptax/cortex-core/src/assets/sprites/"]),
    "ls /home/turfptax/models/":
                            ("What models do I have loaded?",
                             ["ls", "/home/turfptax/models/"]),

    # Date/time
    "date":                 ("What time is it?",
                             ["date"]),
    "date +%Z":             ("What timezone am I in?",
                             ["date", "+%Z"]),
}

# Regex to parse [RUN: <command>] from LLM output
_RUN_PATTERN = re.compile(r"\[RUN:\s*(.+?)\]", re.IGNORECASE)


class BodyShell:
    """Sandboxed shell executor with strict command whitelist."""

    def __init__(self):
        self._cache = {}  # command → (timestamp, output) for short-term caching
        log.info("BodyShell initialized with %d whitelisted commands",
                 len(COMMAND_WHITELIST))

    def get_available_commands(self):
        """Return list of (command_str, description) for prompt injection."""
        return [
            (cmd, info[0]) for cmd, info in COMMAND_WHITELIST.items()
        ]

    def get_command_list_prompt(self):
        """Return a formatted string of available commands for the LLM."""
        lines = ["Available body commands (use [RUN: command] to execute):"]
        for cmd, info in COMMAND_WHITELIST.items():
            lines.append(f"  [RUN: {cmd}] — {info[0]}")
        return "\n".join(lines)

    def execute(self, command_str):
        """Execute a whitelisted command.

        Returns (success: bool, output: str).
        If the command is not whitelisted, returns (False, error_message).
        """
        command_str = command_str.strip()

        if command_str not in COMMAND_WHITELIST:
            log.warning("Blocked non-whitelisted command: %s", command_str)
            return False, f"Command not allowed: {command_str}"

        _, cmd_list = COMMAND_WHITELIST[command_str]

        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip()
            if result.returncode != 0 and result.stderr:
                output = f"{output}\n(stderr: {result.stderr.strip()})"
            # Truncate very long output
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            log.info("Executed: %s → %d bytes output", command_str, len(output))
            return True, output

        except subprocess.TimeoutExpired:
            log.warning("Command timed out: %s", command_str)
            return False, f"Command timed out after 10s: {command_str}"
        except Exception as e:
            log.error("Command failed: %s — %s", command_str, e)
            return False, f"Command error: {e}"

    def execute_multiple(self, command_strs):
        """Execute multiple commands and return combined results.

        Returns list of (command, success, output).
        """
        results = []
        for cmd in command_strs:
            ok, output = self.execute(cmd)
            results.append((cmd, ok, output))
        return results


def parse_run_commands(text):
    """Extract [RUN: command] requests from LLM output.

    Returns list of command strings found in the text.
    """
    return _RUN_PATTERN.findall(text)


def strip_run_commands(text):
    """Remove [RUN: command] tags from text for clean display."""
    return _RUN_PATTERN.sub("", text).strip()


if __name__ == "__main__":
    shell = BodyShell()
    print("Available commands:")
    for cmd, desc in shell.get_available_commands():
        print(f"  {cmd:45s} — {desc}")
    print()
    # Test a few
    for test_cmd in ("uptime", "hostname", "free -m", "rm -rf /"):
        ok, out = shell.execute(test_cmd)
        status = "OK" if ok else "BLOCKED"
        print(f"[{status}] {test_cmd}: {out[:100]}")
