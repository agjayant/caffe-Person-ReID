#!/bin/bash

for name in /home/jayant/vision/Market-1501-v15.09.15/bounding_box_train/*.jpg; do
	convert -resize 128x128\! $name $name
done
