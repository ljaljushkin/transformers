#!/usr/bin/env bash
#echo $PATH
#echo $LD_INCLUDE_PATH
#rm -rf ~/.cache/huggingface/datasets/glue/

HF_SANITY_TEST_OUTPUT_DIR=/home/nlyalyus/sandbox/tmp/develop_hf_patch
export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=$HF_SANITY_TEST_OUTPUT_DIR/repo/transformers
export TORCH_EXTENSIONS_DIR=$HF_SANITY_TEST_OUTPUT_DIR/venv/nncf_extensions
export PATH=$PATH:/usr/local/cuda/bin

python examples/pytorch/text-classification/run_glue.py \
--model_name_or_path roberta-large-mnli \
--task_name mnli \
--do_eval \
--max_seq_length 128 \
--output_dir $HF_SANITY_TEST_OUTPUT_DIR/models/roberta_mnli_PTQ \
--overwrite_output_dir \
--nncf_config nncf_roberta_config_mnli_PTQ.json > roberta_PTQ.log 2>&1 &
#--to_onnx $HF_SANITY_TEST_OUTPUT_DIR/models/roberta_mnli_PTQ/roberta_int8.onnx