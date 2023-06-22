#!/bin/sh
env/bin/uvicorn main:app \
    --workers 3
