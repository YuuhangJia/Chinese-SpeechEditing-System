#!/bin/bash
set -e # set -e 是Shell脚本中的一条设置（set）命令，用于在脚本执行过程中启用“错误即终止”（exit on error）的模式。
NUM_JOB=${NUM_JOB:-36} # NUM_JOB 是一个环境变量；这句话的目的是检查 NUM_JOB 是否已经被设置。如果已经设置，它将使用已经设置的值；如果没有设置，它将使用默认值 36。
echo "| Training MFA using ${NUM_JOB} cores." 
BASE_DIR=data/processed/libritts
MODEL_NAME=${MODEL_NAME:-"mfa_model"}
PRETRAIN_MODEL_NAME=${PRETRAIN_MODEL_NAME:-"mfa_model_pretrain"}
MFA_INPUTS=${MFA_INPUTS:-"mfa_inputs"}
MFA_OUTPUTS=${MFA_OUTPUTS:-"mfa_outputs"}
MFA_CMD=${MFA_CMD:-"train"}
rm -rf $BASE_DIR/mfa_outputs_tmp
if [ "$MFA_CMD" = "train" ]; then
  mfa train $BASE_DIR/$MFA_INPUTS $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/$MODEL_NAME.zip --clean -j $NUM_JOB --config_path data_gen/tts/mfa_train_config.yaml
fi
# rm -rf $BASE_DIR/mfa_tmp $BASE_DIR/$MFA_OUTPUTS
mkdir -p $BASE_DIR/$MFA_OUTPUTS
find $BASE_DIR/mfa_outputs_tmp -regex ".*\.TextGrid" -print0 | xargs -0 -i mv {} $BASE_DIR/$MFA_OUTPUTS/
if [ -e "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_outputs_tmp/unaligned.txt $BASE_DIR/
fi
# rm -rf $BASE_DIR/mfa_outputs_tmp


rm -rf $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_tmp/mfa_inputs_train_acoustic_model/sat_2_ali/textgrids -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
if [ -e "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_outputs_tmp/unaligned.txt $BASE_DIR/
fi
rm -rf $BASE_DIR/mfa_outputs_tmp
rm -rf $BASE_DIR/mfa_tmp