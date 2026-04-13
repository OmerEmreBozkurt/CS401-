#!/bin/bash
# Run hierarchical_analyzer with --louvain-levels 0 (no Louvain) for all 5 datasets
# 3 iterations each, qwen3-coder:480b-cloud

cd "$(dirname "$0")"

run_one() {
    local graph=$1
    local dep_rsf=$2
    local ref_rsf=$3
    local output=$4
    echo ""
    echo "=============================================="
    echo "Running: $graph"
    echo "=============================================="
    python louvian/hierarchical_analyzer.py "$graph" \
        --dependency-rsf "$dep_rsf" \
        --reference-rsf "$ref_rsf" \
        --louvain-resolution 1.0 \
        --model qwen3-coder:480b-cloud \
        --louvain-levels 0 \
        --iterations 3 \
        --output "$output"
}

# Arch (archstudio)
run_one test_results/arch.pkl \
    dataset/archstudio/archstudio-dependency.rsf \
    dataset/archstudio/archstudio-clustering.rsf \
    test_results/arch_hierarchical_results.json

# Bash
run_one test_results/bash-g.pkl \
    dataset/bash/bash-dependency.rsf \
    dataset/bash/bash-clustering.rsf \
    test_results/bash_hierarchical_results.json

# Hadoop
run_one test_results/had.pkl \
    dataset/hadoop/hadoop-dependency.rsf \
    dataset/hadoop/hadoop-clustering.rsf \
    test_results/had_hierarchical_results.json

# ITK
run_one test_results/itk.pkl \
    dataset/itk/itk-dependency.rsf \
    dataset/itk/itk-clustering.rsf \
    test_results/itk_hierarchical_results.json

# Chromium (large - may take 30+ min)
run_one test_results/chromium.pkl \
    dataset/chromium/chromium-dependency.rsf \
    dataset/chromium/chromium-clustering.rsf \
    test_results/chromium_hierarchical_results.json

echo ""
echo "=============================================="
echo "EXTRACTING RESULTS"
echo "=============================================="
for f in test_results/arch_hierarchical_results.json test_results/bash_hierarchical_results.json test_results/had_hierarchical_results.json test_results/itk_hierarchical_results.json test_results/chromium_hierarchical_results.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" _hierarchical_results.json)
        turbomq=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('metrics',{}).get('turbomq','N/A'))" 2>/dev/null)
        mojo=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('metrics',{}).get('mojo_fm','N/A'))" 2>/dev/null)
        echo "$name: TurboMQ=$turbomq, MoJo-FM=$mojo"
    fi
done
