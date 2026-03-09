#!/bin/bash
# run_pipeline.sh - End-to-end execution script for Qwen3.5 9B -> 0.8B distillation
# Hardware: H100 80GB (optimized)

set -e  # Exit immediately if a command exits with a non-zero status

# Text colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Qwen3.5-9B to 0.8B Knowledge Distillation Pipeline (H100) ${NC}"
echo -e "${BLUE}================================================================${NC}\n"

# 1. Verification / Setup
echo -e "${GREEN}[1/4] Setup and Verification...${NC}"
if ! python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required'" &> /dev/null; then
    echo -e "${RED}Error: CUDA not available. Hardware accelerator is required.${NC}"
    exit 1
fi
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo "Detected GPU: $GPU_NAME"
echo ""

# 2. Data Generation Phase
echo -e "${GREEN}[2/4] Generating SFT Data from Teacher (Qwen3.5-9B-Instruct)...${NC}"
echo "This will generate responses for 15,000 prompts (Math, General, Coding)."
echo "Press Ctrl+C to abort, or it will resume if previously interrupted."
python generate_sft_data.py
echo -e "${GREEN}✓ Data generation complete.${NC}\n"

# 3. Phase 1: SFT (Warm-up)
echo -e "${GREEN}[3/4] Running Phase 1: Off-policy SFT (Warm-up)...${NC}"
echo "Training student (Qwen3.5-0.8B-Base) on teacher-generated data."
python train_sft.py
echo -e "${GREEN}✓ Phase 1 SFT complete.${NC}\n"

# 4. Phase 2: KL Distillation
echo -e "${GREEN}[4/4] Running Phase 2: On-policy KL Distillation...${NC}"
echo "Aligning student's reasoning with teacher using Forward KL divergence."
python train_distill.py
echo -e "${GREEN}✓ Phase 2 Distillation complete.${NC}\n"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PIPELINE FINISHED SUCCESSFULLY!${NC}"
echo -e "${BLUE}  Final distilled model saved to: ./output/distilled_0.8b/${NC}"
echo -e "${BLUE}================================================================${NC}"
