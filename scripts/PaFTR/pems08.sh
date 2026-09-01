model_name=PaFTR

root_path_name=./dataset/
data_path_name=PEMS08.npz
model_id_name=PEMS08
data_name=PEMS

seq_len=96
enc_in=170
cycle_len=288

d_model=512
dropout=0
use_revin=1

batch_size=32
lr=0.003
train_epochs=30
patience=5
lradj=type3

use_seq_cycle_complex='complex'
fusion_type='freq'

for random_seed in 2026 2027 2028
do
for pred_len in 12 24 48 96
do
  python -u run.py \
      --is_training 1 \
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
      --random_seed $random_seed
done
done
