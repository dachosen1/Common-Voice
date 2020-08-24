#!/usr/bin/env bash
export IS_DEBUG=${DEBUG:-false}
exec gunicorn run_app:app --log-file - --access-logfile