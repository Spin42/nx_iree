"""List or stage individual hunks of a file's working-tree diff.

  hunks-stage.py <repo> list <file>            print numbered hunks (first lines)
  hunks-stage.py <repo> stage <file> <i,j,..>  stage only those hunks (git apply --cached)
  hunks-stage.py <repo> show <file> <i>        print one hunk in full
"""
import subprocess
import sys

repo, cmd, path = sys.argv[1], sys.argv[2], sys.argv[3]
diff = subprocess.run(
    ["git", "-C", repo, "diff", "-U1", "--", path], capture_output=True, text=True, check=True
).stdout
lines = diff.splitlines(keepends=True)
header, hunks, cur = [], [], None
for ln in lines:
    if ln.startswith("@@"):
        cur = [ln]
        hunks.append(cur)
    elif cur is None:
        header.append(ln)
    else:
        cur.append(ln)

if cmd == "list":
    for i, h in enumerate(hunks):
        changed = [l for l in h[1:] if l[:1] in "+-"]
        print(f"[{i}] {h[0].strip()} ({len(changed)} lines)")
        for l in changed[:4]:
            print("     " + l.rstrip()[:110])
elif cmd == "show":
    sys.stdout.write("".join(hunks[int(sys.argv[4])]))
elif cmd == "stage":
    pick = [int(x) for x in sys.argv[4].split(",")]
    patch = "".join(header) + "".join("".join(hunks[i]) for i in pick)
    subprocess.run(
        ["git", "-C", repo, "apply", "--cached", "--recount", "-"], input=patch, text=True, check=True
    )
    print(f"staged hunks {pick} of {path}")
