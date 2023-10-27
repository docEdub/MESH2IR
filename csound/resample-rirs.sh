DIR=../evaluate/Output/embeddings_0/S1

for FILE in $DIR/*.wav; do src_conv -r48000 -Q5 $FILE -o ${DIR}_48k/$(basename $FILE); done
