#!/bin/bash
# monitor.sh — 每15秒汇报实验进度，完成后打印结果
PID=${1:-40857}
LOG="results/agent_run2_20260510_193610.log"
START=$(date +%s)

echo "监控 PID=$PID，每15秒汇报一次 (Ctrl+C 退出监控，不影响实验)"
echo "-----------------------------------------------------------"

while ps -p "$PID" > /dev/null 2>&1; do
    NOW=$(date +%s)
    ELAPSED=$(( (NOW - START) / 60 ))
    SOCKETS=$(ss -p 2>/dev/null | grep "pid=$PID" | wc -l)
    LOGSIZE=$(wc -c < "$LOG" 2>/dev/null || echo 0)
    CSV_COUNT=$(ls results/experiment_*.csv 2>/dev/null | wc -l)

    echo "[$(date '+%H:%M:%S')] 运行中 | 已等待${ELAPSED}min | active sockets=${SOCKETS} | log=${LOGSIZE}B | csv文件数=${CSV_COUNT}"
    sleep 15
done

echo ""
echo "=============================================="
echo "✅  实验完成！$(date '+%H:%M:%S')"
echo "=============================================="
echo ""
echo "--- 最新 CSV 文件 ---"
ls -la results/experiment_*.csv | tail -5
echo ""
echo "--- 实验日志末尾 ---"
tail -60 "$LOG" 2>/dev/null || echo "(日志为空，输出可能在 stdout)"
