#!/bin/bash
#SBATCH --account=grana_pbl
#SBATCH --partition=all_usr_prod
#SBATCH --output=res_[[GEN_DIFF_FOC_GLOB]].out
#SBATCH --error=res_[[GEN_DIFF_FOC_GLOB]].err
#SBATCH --job-name=[[GEN_DIFF_FOC_GLOB]]
#SBATCH --time=8:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# Run the script with the appropriate arguments

echo "🔹 Job iniziato su $(hostname)"
echo "🔹 Data e ora: $(date)"
echo "🔹 GPU disponibile: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'Nessuna GPU disponibile!')"

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate dsmil
python /work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/nefro_resnet.py

echo "✅ Job completato con successo!"