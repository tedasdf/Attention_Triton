

python eval/regression_gate.py \
  --baseline_summary eval/results/run_0001/summary.json \
  --current_summary eval/results/run_0002/summary.json \
  --max_compile_rate_drop 0.02 \
  --max_pass_at_1_drop 0.01 \
  --output_path eval/results/run_0002/regression_report.json


  python eval/regression_gate.py \
  --baseline_summary eval/results/run_0001/summary.json \
  --current_summary eval/results/run_0002/summary.json \
  --max_compile_rate_drop 0.02 \
  --max_pass_at_1_drop 0.01 \
  --min_compile_rate 0.80 \
  --min_pass_at_1 0.15