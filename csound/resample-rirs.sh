DIR=../evaluate/Output/embeddings_0/S1

mkdir -p ${DIR}_48k

for FILE in $DIR/*.wav; do src_conv -r48000 -Q5 $FILE -o ${DIR}_48k/$(basename $FILE); done
