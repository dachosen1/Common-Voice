export IS_DEBUG=${DEBUG:-false}
exec gunicorn --bind "0.0.0.0:${PORT:-5000}" -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 run_app:app


