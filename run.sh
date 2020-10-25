export IS_DEBUG=${DEBUG:-false}
exec modprobe snd-aloop | gunicorn --bind "0.0.0.0:${PORT:-8080}" --access-logfile - --error-logfile - run_app:app
