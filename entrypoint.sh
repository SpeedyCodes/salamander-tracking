#!/bin/sh

flask --app server.app:app db upgrade

exec "$@"