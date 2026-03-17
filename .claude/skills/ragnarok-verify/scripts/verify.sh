#!/usr/bin/env bash
# ragnarok-verify: Validate ragnarok generation output artifacts.
#
# Usage:
#   bash verify.sh <artifact-path> [--trec25]
#
# --trec25: Enable TREC RAG 2025 specific checks (word limit, docid format, metadata).

set -euo pipefail

ARTIFACT_PATH="${1:?Usage: verify.sh <artifact-path> [--trec25]}"
TREC25=false
if [[ "${2:-}" == "--trec25" ]]; then
  TREC25=true
fi

# Colors (respect NO_COLOR)
if [[ -z "${NO_COLOR:-}" ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; NC=''
fi

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; FAILURES=$((FAILURES + 1)); }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

FAILURES=0

# --- Basic file checks ---
echo "=== File Integrity ==="

if [[ ! -f "$ARTIFACT_PATH" ]]; then
  fail "File not found: $ARTIFACT_PATH"
  exit 1
fi
pass "File exists: $ARTIFACT_PATH"

LINE_COUNT=$(wc -l < "$ARTIFACT_PATH" | tr -d ' ')
if [[ "$LINE_COUNT" -eq 0 ]]; then
  fail "File is empty"
  exit 1
fi
pass "File has $LINE_COUNT records"

# Check every line is valid JSON
BAD_LINES=$(python3 -c "
import json, sys
bad = 0
for i, line in enumerate(open('$ARTIFACT_PATH'), 1):
    line = line.strip()
    if not line:
        continue
    try:
        json.loads(line)
    except json.JSONDecodeError:
        print(f'  Line {i}: invalid JSON', file=sys.stderr)
        bad += 1
print(bad)
")
if [[ "$BAD_LINES" -eq 0 ]]; then
  pass "All lines are valid JSON"
else
  fail "$BAD_LINES lines have invalid JSON"
fi

# --- Content validation ---
echo ""
echo "=== Content Validation ==="

python3 -c "
import json, sys, re

path = '$ARTIFACT_PATH'
trec25 = $( [[ "$TREC25" == "true" ]] && echo "True" || echo "False" )
failures = 0

records = []
for line in open(path):
    line = line.strip()
    if line:
        records.append(json.loads(line))

# Duplicate topic_id check
tids = [r.get('topic_id') or r.get('narrative_id') for r in records]
tids_clean = [t for t in tids if t is not None]
dupes = len(tids_clean) - len(set(tids_clean))
if dupes > 0:
    print(f'✗ {dupes} duplicate topic_id(s)')
    failures += 1
else:
    print(f'✓ No duplicate topic_ids ({len(tids_clean)} unique)')

for i, r in enumerate(records):
    tid = r.get('topic_id') or r.get('narrative_id', f'record-{i+1}')

    # Required fields
    if 'answer' not in r:
        print(f'✗ Record {tid}: missing answer array')
        failures += 1
        continue
    if not isinstance(r['answer'], list) or len(r['answer']) == 0:
        print(f'✗ Record {tid}: empty or non-array answer')
        failures += 1
        continue

    refs = r.get('references', [])

    # Citation integrity
    for j, sent in enumerate(r['answer']):
        if not isinstance(sent, dict) or 'text' not in sent:
            print(f'✗ Record {tid}, sentence {j+1}: missing text field')
            failures += 1
            continue
        for cit in sent.get('citations', []):
            if not isinstance(cit, int) or cit < 0 or cit >= len(refs):
                print(f'✗ Record {tid}, sentence {j+1}: citation index {cit} out of range (0..{len(refs)-1})')
                failures += 1

    # Response length
    word_count = sum(len(s.get('text', '').split()) for s in r['answer'] if isinstance(s, dict))
    declared = r.get('response_length', word_count)
    if abs(word_count - declared) > 5:
        print(f'⚠ Record {tid}: declared response_length={declared}, actual word count={word_count}')

    # TREC 2025 specific checks
    if trec25:
        if word_count > 400:
            print(f'✗ Record {tid}: {word_count} words exceeds 400-word limit')
            failures += 1
        if len(refs) > 100:
            print(f'✗ Record {tid}: {len(refs)} references exceeds 100 limit')
            failures += 1
        msmarco_pat = re.compile(r'^msmarco_v2\.1_doc_\d+_\d+#\d+_\d+$')
        for ref in refs:
            if not msmarco_pat.match(str(ref)):
                print(f'✗ Record {tid}: invalid MS MARCO v2.1 docid: {ref}')
                failures += 1
                break
        for field in ('team_id', 'run_id', 'narrative_id', 'type'):
            if field not in r:
                print(f'✗ Record {tid}: missing TREC metadata field: {field}')
                failures += 1

if failures == 0:
    print('✓ All records are well-formed')
sys.exit(1 if failures > 0 else 0)
" 2>&1 || FAILURES=$((FAILURES + 1))

# --- Summary ---
echo ""
echo "=== Summary ==="
if [[ "$FAILURES" -eq 0 ]]; then
  pass "All checks passed"
  exit 0
else
  fail "$FAILURES check(s) failed"
  exit 1
fi
