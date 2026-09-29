#!/usr/bin/env bash
# ssh into the current box; args are the remote command.
source "$(dirname "$0")/box.env"
exec ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=20 -o LogLevel=ERROR -p "$SSH_PORT" root@"$SSH_HOST" "$@"
