#!/usr/bin/env bash

cd ..

input_data_dir="./Data"
onmt_text_data_dir="./model_input/"

if [ ! -d ${onmt_text_data_dir} ];
then
  mkdir -p "${onmt_text_data_dir}"

  for subset in "train" "valid"; do
      python3 -m dataflow.onmt_helpers.create_onmt_text_data \
          --dialogues_jsonl ${input_data_dir}/${subset}.dataflow_dialogues.jsonl \
          --num_context_turns 2 \
          --include_program \
          --onmt_text_data_outbase ${onmt_text_data_dir}/${subset}
  done
else
  echo "model input already processed"
fi
