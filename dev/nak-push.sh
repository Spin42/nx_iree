#!/usr/bin/env bash
# Copy the NAK sources edited in the host tree into Buildroot's Mesa, rebuild
# the aarch64 driver, and push it to /root/nvkexp on the board.
# Never run while another Mesa build is in progress.
set -euo pipefail
A=/var/home/marc/Projects/nerves_system_okdo_c100/.claude/worktrees/base-system/.nerves/artifacts/nerves_system_okdo_c100-portable-0.1.0
T=/var/home/marc/.claude/jobs/7dc3895b/tmp
NAK=src/nouveau/compiler/nak
for f in opt_instr_sched_common.rs opt_instr_sched_prepass.rs opt_instr_sched_postpass.rs; do
  cp $T/mesa-host-src/$NAK/$f $A/build/mesa3d-26.1.8/$NAK/$f
done
make -C "$A" mesa3d-rebuild > "$T/mesa-rebuild.log" 2>&1 || { grep -a -E "error(\[|:)|ERROR" "$T/mesa-rebuild.log" | head -30; exit 1; }
"$A/host/bin/aarch64-nerves-linux-gnu-strip" -o "$T/libvulkan_nouveau.so" "$A/target/usr/lib/libvulkan_nouveau.so"
cat > "$T/nvk-dev.json" <<'EOF'
{"ICD": {"api_version": "1.4.354", "library_arch": "64", "library_path": "/root/nvkexp/libvulkan_nouveau.so"}, "file_format_version": "1.0.1"}
EOF
timeout 20 ssh -o BatchMode=yes -o LogLevel=ERROR 192.168.0.31 'File.mkdir_p!("/root/nvkexp"); :ok' > /dev/null
printf 'put %s /root/nvkexp/libvulkan_nouveau.so\nput %s /root/nvkexp/nvk-dev.json\n' \
  "$T/libvulkan_nouveau.so" "$T/nvk-dev.json" | timeout 120 sftp -q -o BatchMode=yes -o LogLevel=ERROR -b - 192.168.0.31 > /dev/null
echo "pushed $(stat -c %s $T/libvulkan_nouveau.so) bytes"
