#!/bin/bash
set -e # 脚本中任何命令执行失败时立即退出脚本
NUM_JOB=${NUM_JOB:-36} # 获取进程数
echo "| Training MFA using ${NUM_JOB} cores."


BASE_DIR=data/processed/aishell3
MODEL_NAME=${MODEL_NAME:-"mfa_model"}
PRETRAIN_MODEL_NAME=${PRETRAIN_MODEL_NAME:-"mfa_model_pretrain"}
MFA_INPUTS=${MFA_INPUTS:-"mfa_inputs"}
MFA_OUTPUTS=${MFA_OUTPUTS:-"mfa_outputs"}
MFA_CMD=${MFA_CMD:-"train"}

# 获取脚本的完整路径
script_path=$(readlink -f "$0")
# 提取路径部分，即脚本所在的目录
script_dir=$(dirname "$script_path")

rm -rf $BASE_DIR/mfa_outputs_tmp # 这个必须要，暂时放Output，但这里面实际上什么都不会生成
mkdir -p $BASE_DIR/mfa_outputs_tmp

# 下面这一步生成，mfa_dict.txt文件以及一些临时文件在\mfa_tmp\文件夹下：
echo "mfa train ${BASE_DIR}/${MFA_INPUTS} ${BASE_DIR}/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t ${BASE_DIR}/mfa_tmp -o ${BASE_DIR}/${MODEL_NAME}.zip --clean -j ${NUM_JOB} --config_path ${BASE_DIR}/mfa_train_config.yaml"
if [ "$MFA_CMD" = "train" ]; then
  mfa train $BASE_DIR/$MFA_INPUTS $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/$MODEL_NAME.zip --clean -j $NUM_JOB --config_path $script_dir/mfa_train_config.yaml
fi

# 下面的步骤在mfa_outputs文件夹下生成最终的.TextGrid文件
rm -rf $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_tmp/mfa_inputs_train_acoustic_model/sat_2_ali/textgrids -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
if [ -e "$BASE_DIR/mfa_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_tmp/unaligned.txt $BASE_DIR/
fi
rm -rf $BASE_DIR/mfa_tmp