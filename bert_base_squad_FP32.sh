#!/usr/bin/env bash
#echo $PATH
#echo $LD_INCLUDE_PATH
#rm -rf ~/.cache/huggingface/datasets/glue/

HF_SANITY_TEST_OUTPUT_DIR=/home/nlyalyus/sandbox/tmp/develop_hf_patch
export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=$HF_SANITY_TEST_OUTPUT_DIR/repo/transformers
export TORCH_EXTENSIONS_DIR=$HF_SANITY_TEST_OUTPUT_DIR/venv/nncf_extensions
export PATH=$PATH:/usr/local/cuda/bin
MODEL_NAME=vs_bert_base_squad
NNCF_CONFIG_NAME=nncf_bert_config_squad_PTQ

python examples/pytorch/question-answering/run_qa.py \
--model_name_or_path vuiseng9/bert-base-uncased-squad \
--dataset_name squad \
--do_eval \
--per_device_eval_batch_size 128 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir $HF_SANITY_TEST_OUTPUT_DIR/models/$MODEL_NAME \
--overwrite_output_dir
#--nncf_config $NNCF_CONFIG_NAME.json > $MODEL_NAME.log 2>&1 &
#--to_onnx $HF_SANITY_TEST_OUTPUT_DIR/models/$MODEL_NAME/model.onnx