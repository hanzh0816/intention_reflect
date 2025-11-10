#!/bin/bash
# 查找最近训练的checkpoints

EXPERIMENT_ROOT="/data2/hzh/nuplan/exp/planTF"

echo "=== 查找PlanTF训练的checkpoints ==="
echo ""

if [ ! -d "$EXPERIMENT_ROOT" ]; then
    echo "实验目录不存在: $EXPERIMENT_ROOT"
    exit 1
fi

# 找到最新的实验目录
LATEST_EXP=$(find $EXPERIMENT_ROOT -maxdepth 1 -type d -name "20*" | sort -r | head -1)

if [ -z "$LATEST_EXP" ]; then
    echo "未找到任何实验"
    exit 1
fi

echo "最新实验目录: $LATEST_EXP"
echo ""

# 查找checkpoints
CHECKPOINT_DIR="$LATEST_EXP/checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "可用的checkpoints:"
echo "-------------------"
ls -lh "$CHECKPOINT_DIR"/*.ckpt 2>/dev/null | awk '{print $9, "(" $5 ")"}'

echo ""
echo "=== 推荐使用的checkpoint ==="

# 找到最佳的checkpoint（基于val_minFDE）
BEST_CKPT=$(ls "$CHECKPOINT_DIR"/epoch=*-val_minFDE*.ckpt 2>/dev/null | sort -t'=' -k3 -n | head -1)

if [ -n "$BEST_CKPT" ]; then
    echo "最佳checkpoint (最低val_minFDE): $BEST_CKPT"
else
    # 如果没有val_minFDE，找last.ckpt
    LAST_CKPT=$(ls "$CHECKPOINT_DIR"/last.ckpt 2>/dev/null)
    if [ -n "$LAST_CKPT" ]; then
        echo "Last checkpoint: $LAST_CKPT"
        BEST_CKPT=$LAST_CKPT
    else
        echo "未找到任何checkpoint"
        exit 1
    fi
fi

echo ""
echo "=== 快速测试命令 ==="
echo "开环测试:"
echo "  sh ./test_open_loop.sh $BEST_CKPT"
echo ""
echo "闭环测试:"
echo "  sh ./test_closed_loop.sh $BEST_CKPT test14-random"
