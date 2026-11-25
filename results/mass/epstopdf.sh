#!/bin/sh

find ./ -name *.eps >! file

while read line
do
    echo $line
    ps2pdf -dEPSCrop $line
done < file

find ./ -name *.pdf >! file
