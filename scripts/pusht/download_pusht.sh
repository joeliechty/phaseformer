#!/bin/sh
# Download the Diffusion Policy Push-T demonstrations (Chi et al. 2023, RSS).
#
#   scripts/pusht/download_pusht.sh                    # -> <repo>/data/pusht  (git-ignored)
#   scripts/pusht/download_pusht.sh /scratch/pusht     # explicit destination
#   PUSHT_DATA_DIR=/scratch/pusht scripts/pusht/download_pusht.sh
#   scripts/pusht/download_pusht.sh --dry-run          # show what would happen, touch nothing
#   scripts/pusht/download_pusht.sh --force            # refetch even if already present
#
# No path is hardcoded: the default destination is resolved relative to this script's own
# location, so the repo can live anywhere. ~30 MB; unpacks to a zarr store of the same size.
set -eu

URL="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZIP_BYTES=30988725            # verified 2026-09; a mismatch means the artifact moved or changed
STORE="pusht_cchi_v7_replay.zarr"
: "${PUSHT_ZIP_SHA256:=63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9}"  # digest pin; set empty to enforce byte size only

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

DRY=0; FORCE=0; DEST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --force)   FORCE=1 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    -*)        echo "unknown flag: $1" >&2; exit 2 ;;
    *)         DEST="$1" ;;
  esac
  shift
done
DEST=${DEST:-${PUSHT_DATA_DIR:-$REPO_ROOT/data/pusht}}

say() { printf '%s\n' "$*"; }

# If the destination is inside the repo, it must be git-ignored -- 30 MB of demonstrations does
# not belong in the history.
case "$DEST" in
  "$REPO_ROOT"/*)
    if ! (cd "$REPO_ROOT" && git check-ignore -q "$DEST") 2>/dev/null; then
      say "WARNING: $DEST is inside the repo but is NOT git-ignored. Add it to .gitignore."
    fi ;;
esac

if [ -d "$DEST/$STORE" ] && [ "$FORCE" -eq 0 ]; then
  say "already present: $DEST/$STORE   (--force to refetch)"
  exit 0
fi

ZIP="$DEST/pusht.zip"
if [ "$DRY" -eq 1 ]; then
  say "DRY: mkdir -p $DEST"
  say "DRY: fetch $URL -> $ZIP"
  say "DRY: verify $ZIP_BYTES bytes${PUSHT_ZIP_SHA256:+ and sha256 $PUSHT_ZIP_SHA256}"
  say "DRY: unzip and move pusht/$STORE -> $DEST/$STORE"
  exit 0
fi

mkdir -p "$DEST"
if [ ! -f "$ZIP" ] || [ "$FORCE" -eq 1 ]; then
  say "fetching $URL  (~30 MB)"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --continue-at - -o "$ZIP" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$ZIP" "$URL"
  else
    say "need curl or wget" >&2; exit 1
  fi
fi

GOT=$(wc -c < "$ZIP" | tr -d ' ')
if [ "$GOT" != "$ZIP_BYTES" ]; then
  say "size mismatch: got $GOT bytes, expected $ZIP_BYTES" >&2
  say "the upstream artifact may have changed; verify before trusting it" >&2
  exit 1
fi

SUM=""
if   command -v shasum    >/dev/null 2>&1; then SUM=$(shasum -a 256 "$ZIP" | cut -d' ' -f1)
elif command -v sha256sum >/dev/null 2>&1; then SUM=$(sha256sum "$ZIP" | cut -d' ' -f1)
fi
if [ -n "$SUM" ]; then
  say "sha256 $SUM"
  if [ -n "$PUSHT_ZIP_SHA256" ] && [ "$SUM" != "$PUSHT_ZIP_SHA256" ]; then
    say "sha256 mismatch: expected $PUSHT_ZIP_SHA256" >&2; exit 1
  fi
fi

# The zip carries a top-level pusht/ directory; flatten the store into $DEST.
TMP=$(mktemp -d "$DEST/.unzip.XXXXXX")
trap 'rm -rf "$TMP"' EXIT INT TERM
unzip -q "$ZIP" -d "$TMP"
rm -rf "$DEST/$STORE"
mv "$TMP/pusht/$STORE" "$DEST/$STORE"

say "ready: $DEST/$STORE"
say "next:  python scripts/pusht/convert_pusht.py --zarr \"$DEST/$STORE\""

# The LeRobot mirror (huggingface.co/datasets/lerobot/pusht) is deliberately not scripted: its
# observation.state is 2-D (agent position only) and drops the T-block pose, so it cannot
# support the SE(2) object state this project is built around.
