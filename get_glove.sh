#!/usr/bin/env bash

glove_840b_dir="output/glove_840b"
if [ ! -d ${glove_840b_dir} ];
then
    mkdir -p "${glove_840b_dir}"
    wget -O ${glove_840b_dir}/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip ${glove_840b_dir}/glove.840B.300d.zip -d ${glove_840b_dir}
else
    echo "glove already downloaded"
fi
