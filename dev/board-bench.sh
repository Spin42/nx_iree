#!/usr/bin/env bash
# Pack the board bundle, upload it, run jetson/run.exs on the board and print the log.
# NXIREE_BOARD_ENV / NXIREE_DUMP_OUT are forwarded to board_run.exs; with
# NXIREE_DUMP_OUT the dump is fetched to $T/<basename>.
set -e
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
R=/var/home/marc/Projects/nx_iree/bench/jetson
cp $R/run.exs $T/nxiree-board/run.exs
cp $R/board_run.exs $T/scripts/nxiree-board-run.exs
tar czf $T/nxiree-board.tgz -C $T/nxiree-board .
sftp -o BatchMode=yes -o LogLevel=ERROR -b $T/sftp-nxiree 192.168.0.31 > /dev/null
ssh -o BatchMode=yes -o LogLevel=ERROR 192.168.0.31 'File.rm_rf!("/root/nxiree"); :ok = :erl_tar.extract(~c"/root/nxiree-board.tgz", [:compressed, cwd: ~c"/root/nxiree"]); :ok' > /dev/null
PRE="System.delete_env(\"NXIREE_BOARD_ENV\"); System.delete_env(\"NXIREE_DUMP_OUT\"); System.delete_env(\"NXIREE_PIN_HZ\");"
[ -n "${NXIREE_BOARD_ENV:-}" ] && PRE="$PRE System.put_env(\"NXIREE_BOARD_ENV\", \"$NXIREE_BOARD_ENV\");"
[ -n "${NXIREE_DUMP_OUT:-}" ] && PRE="$PRE System.put_env(\"NXIREE_DUMP_OUT\", \"$NXIREE_DUMP_OUT\");"
[ -n "${NXIREE_PIN_HZ:-}" ] && PRE="$PRE System.put_env(\"NXIREE_PIN_HZ\", \"$NXIREE_PIN_HZ\");"
timeout 900 ssh -o BatchMode=yes -o LogLevel=ERROR 192.168.0.31 "$PRE Code.eval_file(\"/root/nxiree-board-run.exs\"); :ok" 2>&1 \
  | grep -vE "WARNING: NVK|^:ok|^\{:ok" | grep -E "compiled for|matmul|chain|tiny|exit|wrote|TRAP|ILLEGAL|fault|Error|error|\*\*"
if [ -n "${NXIREE_DUMP_OUT:-}" ]; then
  printf 'get %s %s/%s\n' "$NXIREE_DUMP_OUT" "$T" "$(basename $NXIREE_DUMP_OUT)" | sftp -q -o BatchMode=yes -o LogLevel=ERROR -b - 192.168.0.31 > /dev/null
  ls -la $T/$(basename $NXIREE_DUMP_OUT)
fi
