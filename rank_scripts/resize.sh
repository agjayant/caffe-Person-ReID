#!/bin/bash

for name in /home/jayant/caffe-Person-ReID_triplet/rank_scripts/images_market/*.jpg; do
	convert -resize 128x256\! $name $name
done
