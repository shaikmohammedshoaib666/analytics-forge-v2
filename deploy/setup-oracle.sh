#!/usr/bin/env bash
# Back-compat wrapper — prefer deploy/setup-vm.sh (works on Azure + Oracle).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/setup-vm.sh" "$@"
