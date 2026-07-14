#!/usr/bin/env bash
#
# Pre-commit hygiene gate — de-identification token scan over the staged tracked diff.
#
# Blocks a commit whose staged additions would introduce a barred identity, tooling,
# or internal-shorthand token onto the tracked (public) surface. The barred-token list
# is kept in a local, untracked file (.hygiene_tokens.local) so the denylist itself —
# which necessarily spells out the tokens it bars — never reaches the public tree; see
# the README "Local setup" note for re-creating it on a fresh working machine.
#
# The one sanctioned exception is the named third-party facility citation: the Bloomberg
# analyst-data override (BE998 / PX395) and the "UBSG" market-data ticker. Those are the
# only permitted occurrences of "ubs" and are allow-listed below by their citation context.
#
# Scope — two modes:
#   (default)            staged ADDED lines only. Catches NEW violations at commit time;
#                        pre-existing content elsewhere in a file is not re-flagged, so
#                        residue that predates the gate is structurally invisible to it.
#   --full-tree / --all  every tracked file's current content. The separate, necessary
#                        check that catches pre-existing residue; run it before every
#                        merge/push to a public main.
# Three files are excluded from either scan: this gate (it carries the sanctioned
# citations), the local Claude Code config dir, and the local token list.
#
set -euo pipefail

mode="staged"
case "${1:-}" in
  --full-tree|--all) mode="full" ;;
  "") ;;
  *) echo "usage: hygiene_gate.sh [--full-tree|--all]" >&2; exit 2 ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
token_file="$repo_root/.hygiene_tokens.local"

# Pathspecs never scanned. (.claude and the token file are ignored and so never staged;
# excluding them is belt-and-suspenders. The gate script itself must be excluded because
# it holds the sanctioned citation strings.)
exclude=(':!scripts/hygiene_gate.sh' ':!.claude/**' ':!.hygiene_tokens.local')

if [[ ! -f "$token_file" ]]; then
  echo "hygiene-gate: token list is missing at $token_file." >&2
  echo "  Re-create it (see README 'Local setup') before committing; refusing to pass blind." >&2
  exit 1
fi

if [[ "$mode" == "full" ]]; then
  # Full-tree corpus: every tracked file's content (git grep over the working tree,
  # tracked paths only, binaries skipped). Hits report as file:line.
  scan() { git -C "$repo_root" grep -inIE "$1" -- . "${exclude[@]}" || true; }
  scope_label="tracked tree"
else
  # Staged additions across tracked files (added lines only; drop the +++ file headers).
  staged="$(git diff --cached --no-color -U0 --diff-filter=ACM -- . "${exclude[@]}" \
    | grep -E '^\+' | grep -Ev '^\+\+\+' || true)"

  if [[ -z "$staged" ]]; then
    exit 0
  fi
  scan() { printf '%s\n' "$staged" | grep -inE "$1" || true; }
  scope_label="staged additions"
fi

fail=0

# 1) Hard denylist from the local token file (one grep -iE pattern per non-comment line).
while IFS= read -r pat || [[ -n "$pat" ]]; do
  [[ -z "${pat// }" || "$pat" == \#* ]] && continue
  hits="$(scan "$pat")"
  if [[ -n "$hits" ]]; then
    echo "hygiene-gate: barred token /$pat/ in $scope_label:" >&2
    printf '%s\n' "$hits" | sed 's/^/    /' >&2
    fail=1
  fi
done < "$token_file"

# 2) 'ubs' is permitted ONLY inside the sanctioned Bloomberg citations. Word-boundary the
#    match (so subset/substr do not trip it), then subtract the citation contexts; any
#    residue is an unsanctioned occurrence and blocks.
ubs_hits="$(scan '\bubs\b' \
  | grep -ivE 'BE998|PX395|UBSG|BEST_ANALYST_REC|BEST_TARGET_PRICE|INTERVAL_END_VALUE_DATE|Best Analyst Rating' \
  || true)"
if [[ -n "$ubs_hits" ]]; then
  echo "hygiene-gate: unsanctioned 'ubs' occurrence (only BE998=UBS / PX395 / UBSG citations are allowed):" >&2
  printf '%s\n' "$ubs_hits" | sed 's/^/    /' >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "hygiene-gate: BLOCKED. Remove the tokens above, or — if a genuine sanctioned" >&2
  echo "  third-party citation — add its context to the allow-list in this gate explicitly." >&2
  exit 1
fi

exit 0
