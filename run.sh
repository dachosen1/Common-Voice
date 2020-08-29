#!/usr/bin/env bash
export IS_DEBUG=${DEBUG:-false}
## shellcheck disable=SC2155
#export UID=$(id -u)
## shellcheck disable=SC2155
#export GID=$(id -g)
#docker build --build-arg USER="$USER" \
#             --build-arg UID="$UID" \
#             --build-arg GID="$GID" \
#             --build-arg PW=<PASSWORD IN CONTAINER> \
#             -t common-voice-app \
#             -f common-voice \
#             .
exec gunicorn --bind 0.0.0.0:5000 --access-logfile - --error-logfile - run_app:app