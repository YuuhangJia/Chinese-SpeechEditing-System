#!/bin/bash
set -e # �ű����κ�����ִ��ʧ��ʱ�����˳��ű�
NUM_JOB=${NUM_JOB:-36} # ��ȡ������
echo "| Training MFA using ${NUM_JOB} cores."


BASE_DIR=data/processed/aishell3
MODEL_NAME=${MODEL_NAME:-"mfa_model"}
PRETRAIN_MODEL_NAME=${PRETRAIN_MODEL_NAME:-"mfa_model_pretrain"}
MFA_INPUTS=${MFA_INPUTS:-"mfa_inputs"}
MFA_OUTPUTS=${MFA_OUTPUTS:-"mfa_outputs"}
MFA_CMD=${MFA_CMD:-"train"}

# ��ȡ�ű�������·��
script_path=$(readlink -f "$0")
# ��ȡ·�����֣����ű����ڵ�Ŀ¼
script_dir=$(dirname "$script_path")

rm -rf $BASE_DIR/mfa_outputs_tmp # �������Ҫ����ʱ��Output����������ʵ����ʲô����������
mkdir -p $BASE_DIR/mfa_outputs_tmp

# ������һ�����ɣ�mfa_dict.txt�ļ��Լ�һЩ��ʱ�ļ���\mfa_tmp\�ļ����£�
echo "mfa train ${BASE_DIR}/${MFA_INPUTS} ${BASE_DIR}/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t ${BASE_DIR}/mfa_tmp -o ${BASE_DIR}/${MODEL_NAME}.zip --clean -j ${NUM_JOB} --config_path ${BASE_DIR}/mfa_train_config.yaml"
if [ "$MFA_CMD" = "train" ]; then
  mfa train $BASE_DIR/$MFA_INPUTS $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/$MODEL_NAME.zip --clean -j $NUM_JOB --config_path $script_dir/mfa_train_config.yaml
fi

# ����Ĳ�����mfa_outputs�ļ������������յ�.TextGrid�ļ�
rm -rf $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_tmp/mfa_inputs_train_acoustic_model/sat_2_ali/textgrids -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
if [ -e "$BASE_DIR/mfa_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_tmp/unaligned.txt $BASE_DIR/
fi
rm -rf $BASE_DIR/mfa_tmp