#!/bin/bash

docker build -t nlp_service .
docker run -p 8000:8000 nlp_service
