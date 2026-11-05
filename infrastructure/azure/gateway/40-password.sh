#!/bin/sh
set -eu
: "${STOCKWISE_HTPASSWD:?Password hash must be configured}"
umask 077
printf '%s\n' "$STOCKWISE_HTPASSWD" > /run/stockwise.htpasswd
chmod 644 /run/stockwise.htpasswd
