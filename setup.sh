#!/usr/bin/env bash

PIPPATH="/content/drive/My Drive/6.806 Final Project/6.806-Final-Project/packages"
mkdir -p "${PIPPATH}"

# Install the sm-dataflow package and its core dependencies
pip3 install --target="${PIPPATH}" git+https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis.git

# echo task_oriented_dialogue_flow installed

# # Download the spaCy model for tokenization
# python3 -m --target=${PIPPATH} spacy download en_core_web_md-2.2.0 --direct
# echo spacy and others installed

# # Install OpenNMT-py and PyTorch for training and running the models
# pip3 install --target=${PIPPATH} OpenNMT-py==1.0.0 torch==1.4.0
# echo opennmt installed

# pip3 install --target=${PIPPATH} typing
# echo typing installed

# pip3 install -r --target=${PIPPATH} task_oriented_dialogue_as_dataflow_synthesis/requirements-dev.txt
# echo other reqs installed