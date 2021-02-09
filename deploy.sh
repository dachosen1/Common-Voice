#!/bin/bash

cd ~/circleci-test
git pull origin main
docker-compose up --build