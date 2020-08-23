#!/usr/bin/env bash
exec gunicorn run_app:app --log-file - --access-logfile -