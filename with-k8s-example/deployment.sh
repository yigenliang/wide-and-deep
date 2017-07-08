#!/usr/bin/env bash

sudo rm -rf /shared/model/*
kubectl create -f tf-k8s-working.yaml --validate=true
