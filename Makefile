NAME=common-voice
COMMIT_ID=$(shell git rev-parse HEAD)


build-common-voice-heroku:
	docker build registry.heroku.com/$(NAME)/web:$(COMMIT_ID) .

push-common-voice-heroku:
	docker push registry.heroku.com/${HEROKU_APP_NAME}/web:$(COMMIT_ID)