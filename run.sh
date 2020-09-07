export IS_DEBUG=${DEBUG:-false}
exec gunicorn --bind "0.0.0.0:${PORT:-5000}" --access-logfile - --error-logfile - run_app:app
