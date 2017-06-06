#!/usr/bin/env bash

kubectl delete rc tf-ps0

kubectl delete service tf-ps0 tf-worker0 tf-master0

kubectl delete job tf-worker0 tf-master0

rm -rf ../model/*
