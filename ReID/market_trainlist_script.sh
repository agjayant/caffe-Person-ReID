#!/bin/bash

#The first four characters as label
(ls images_market/ | cut -c-4)> list.txt

#The list of images with full path
cd images_market/
ls -d -1 $PWD/* > ../market.txt

cd ..

#combining the two
(paste -d ' ' <(cut -d ' ' -f 1 market.txt) <(cut -d ' ' -f 1 list.txt))> checklist.txt
rm market.txt list.txt
  



