#!/usr/bin/env bash
# populate_aws_credentials.sh
#
# Reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_SESSION_TOKEN
# from the current shell environment and writes them into
# aws_credentials.json at the repository root.
#
# Usage:
#   bash scripts/bash/populate_aws_credentials.sh

set -euo pipefail

# Locate the repository root relative to this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CREDENTIALS_FILE="${REPO_ROOT}/aws_credentials.json"

# Validate that the required environment variables are set.
missing=()
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
    if [[ -z "${!var:-}" ]]; then
        missing+=("$var")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: The following required environment variables are not set:" >&2
    for var in "${missing[@]}"; do
        echo "  - $var" >&2
    done
    exit 1
fi

# Write the credentials to aws_credentials.json.
cat > "${CREDENTIALS_FILE}" <<EOF
{
  "AWS_ACCESS_KEY_ID": "${AWS_ACCESS_KEY_ID}",
  "AWS_SECRET_ACCESS_KEY": "${AWS_SECRET_ACCESS_KEY}",
  "AWS_SESSION_TOKEN": "${AWS_SESSION_TOKEN}"
}
EOF

echo "AWS credentials written to ${CREDENTIALS_FILE}"
chmod 600 "${CREDENTIALS_FILE}"
