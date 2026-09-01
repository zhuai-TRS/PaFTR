model_name=PaFTR

root_path_name=./dataset/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS

seq_len=96
enc_in=358
cycle_len=288

d_model=512
dropout=0
use_revin=0

batch_size=32
lr=0.003
train_epochs=30
patience=5
lradj=type3

use_seq_cycle_complex='complex'
fusion_type='freq'
random_seed=2026

# Frozen main checkpoint (seed 2026). Clock offsets: ±1/2/3 h = ±12/24/36 steps.
for pred_len in 12 24 48 96
do
  for cycle_shift in -36 -24 -12 12 24 36
  do
    python -u run.py \
        --is_training 0 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id ${model_id_name}_${seq_len}_${pred_len} \
        --model $model_name \
        --data $data_name \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --cycle_len $cycle_len \
        --d_model $d_model \
        --dropout $dropout \
        --use_revin $use_revin \
        --batch_size $batch_size \
        --learning_rate $lr \
        --train_epochs $train_epochs \
        --patience $patience \
        --itr 1 \
        --lradj $lradj \
        --use_seq_cycle_complex $use_seq_cycle_complex \
        --fusion_type $fusion_type \
        --random_seed $random_seed \
        --cycle_shift $cycle_shift
  done
done
