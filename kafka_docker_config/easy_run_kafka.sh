#!/bin/sh
sudo rm /tmp/* -R && sudo docker-compose rm && sudo docker-compose -f docker-compose-single-broker.yml up
