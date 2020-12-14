export IS_DEBUG=${DEBUG:-false}
exec gunicorn --bind "0.0.0.0:${PORT:-8080}" --worker-class eventlet -w 1 run_app:app


