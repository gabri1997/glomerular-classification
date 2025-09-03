#!/bin/bash
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --output=res_[[INTENS]].out
#SBATCH --error=res_[[INTENS]].err
#SBATCH --job-name=[[INTENS]]
#SBATCH --time=10:00:00
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


# Run the script with the appropriate arguments

echo "🔹 Job iniziato su $(hostname)"
echo "🔹 Data e ora: $(date)"
echo "🔹 GPU disponibile: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'Nessuna GPU disponibile!')"

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate dsmil
python /work/grana_far2023_fomo/Pollastri_Glomeruli/Train_scripts/nefro_resnet.py \
--project_name Train_R18_New_Data_SGD_Seed42_RGB_Lr001 --old_or_new_dataset_folder Files/ \
--label INTENS --train_or_test Test_on_folds --val_loss True --scheduler SGD --weights _New22 --conf_matrix_label "0,0.5,1,1.5,2,2.5,3" --classes 7 --wandb_flag True --wloss True --sampler False \
--load_for_fine_tuning False --learning_rate 0.01

echo "✅ Job completato con successo!"
