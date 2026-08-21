#!/usr/bin/env bash
# Dead-man's switch for JarvisLabs instances (Joe, 2026-08-20).
# Pauses Running instances when either:
#   (1) the managing Claude session has gone quiet (stale heartbeat), or
#   (2) the instance is IDLE — no compute process running — for a
#       grace period. Case (2) was added after a crashed job left a
#       box billing for 16 idle hours (2026-08-20): a live-but-idle
#       box is invisible to a heartbeat check alone.
# Pause preserves disk and results; destroy is NEVER automated.
#
# Heartbeat contract: only the active session touches $HB, and only
# when it actually polls or manages runs. Nothing automated may
# touch it — an automated toucher defeats the switch.
#
# crontab: */20 * * * * $HOME/.local/bin/jl_watchdog.sh
set -u
HB="$HOME/.jl_watchdog/heartbeat"
LOG="$HOME/.jl_watchdog/log"
IDLE_STATE="$HOME/.jl_watchdog/idle"
TTL=10800        # 3h without a session heartbeat
IDLE_TTL=2400    # 40m of no compute process on the box
mkdir -p "$HOME/.jl_watchdog" "$IDLE_STATE"

# cron's PATH is /usr/bin:/bin — it does NOT include ~/.local/bin,
# so `command -v jl` came back empty and this script exited silently
# for 16 hours while an instance billed. Search explicitly and log
# loudly if the CLI is genuinely missing.
PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
JL="$(command -v jl || true)"
if [ -z "$JL" ]; then
    echo "$(date -Is) FATAL: jl CLI not found; cannot pause" >> "$LOG"
    exit 1
fi

now=$(date +%s)
hb=$(stat -c %Y "$HB" 2>/dev/null || echo 0)
hb_age=$(( now - hb ))

# machine_id<TAB>ip for every Running instance
"$JL" list --json 2>/dev/null | python3 -c '
import json, sys
try:
    rows = json.load(sys.stdin)
except Exception:
    sys.exit(0)
if not isinstance(rows, list):
    sys.exit(0)
for r in rows:
    if str(r.get("status", "")).lower() != "running":
        continue
    ip = ""
    for k in ("ssh_command", "public_ip", "ip"):
        v = r.get(k)
        if not isinstance(v, str):
            continue
        if "@" in v:
            ip = v.rsplit("@", 1)[-1].split()[0]
            break
        if v.count(".") == 3:
            ip = v
            break
    print(str(r.get("machine_id")) + chr(9) + ip)
' | while IFS=$'\t' read -r mid ip; do
    [ -z "${mid:-}" ] && continue
    reason=""
    if [ "$hb_age" -ge "$TTL" ]; then
        reason="session heartbeat stale ${hb_age}s"
    elif [ -n "${ip:-}" ]; then
        # idle check: any block2/python compute running on the box?
        busy=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
                   -o ConnectTimeout=10 "ubuntu@$ip" \
                   'pgrep -cf "[m]assgap[.]py"  || true' \
                   2>/dev/null | tr -dc '0-9')
        mark="$IDLE_STATE/$mid"
        if [ "${busy:-0}" -gt 0 ]; then
            rm -f "$mark"
        else
            [ -f "$mark" ] || touch "$mark"
            since=$(stat -c %Y "$mark" 2>/dev/null || echo "$now")
            idle_age=$(( now - since ))
            [ "$idle_age" -ge "$IDLE_TTL" ] && \
                reason="idle (no compute) ${idle_age}s"
        fi
    fi
    if [ -n "$reason" ]; then
        echo "$(date -Is) $reason -> pausing $mid" >> "$LOG"
        "$JL" pause "$mid" --yes --json >> "$LOG" 2>&1
        rm -f "$IDLE_STATE/$mid"
    fi
done
