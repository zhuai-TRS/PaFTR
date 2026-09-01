model_name=PaFTR

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

seq_len=96
enc_in=862
cycle_len=168
threshold=48.5

d_model=512
dropout=0
use_revin=1

batch_size=16
lr=0.003
train_epochs=30
patience=5

use_seq_cycle_complex='complex'
fusion_type='freq'
random_seed=2026

# Frozen main checkpoint (seed 2026). Clock offsets: ±1/2/3 h = ±1/2/3 steps.
for pred_len in 96 192 336 720
do
  for cycle_shift in -3 -2 -1 1 2 3
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
        --threshold $threshold \
        --d_model $d_model \
        --dropout $dropout \
        --use_revin $use_revin \
        --batch_size $batch_size \
        --learning_rate $lr \
        --train_epochs $train_epochs \
        --patience $patience \
        --itr 1 \
        --use_seq_cycle_complex $use_seq_cycle_complex \
        --fusion_type $fusion_type \
        --random_seed $random_seed \
        --cycle_shift $cycle_shift
  done
done
