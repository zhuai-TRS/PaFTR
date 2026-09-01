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

qkv='csf'
for pred_len in 12 24 48 96
do
 python -u run.py \
     --is_training 1 \
     --root_path $root_path_name \
     --data_path $data_path_name \
     --model_id ${model_id_name}_${seq_len}_${pred_len}_csf \
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
     --random_seed $random_seed \
     --lradj $lradj \
     --use_seq_cycle_complex $use_seq_cycle_complex \
     --fusion_type $fusion_type \
     --qkv $qkv
done

qkv='fcs'
for pred_len in 12 24 48 96
do
 python -u run.py \
     --is_training 1 \
     --root_path $root_path_name \
     --data_path $data_path_name \
     --model_id ${model_id_name}_${seq_len}_${pred_len}_fcs \
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
     --random_seed $random_seed \
     --lradj $lradj \
     --use_seq_cycle_complex $use_seq_cycle_complex \
     --fusion_type $fusion_type \
     --qkv $qkv
done

qkv='fsc'
for pred_len in 12 24 48 96
do
 python -u run.py \
     --is_training 1 \
     --root_path $root_path_name \
     --data_path $data_path_name \
     --model_id ${model_id_name}_${seq_len}_${pred_len}_fsc \
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
     --random_seed $random_seed \
     --lradj $lradj \
     --use_seq_cycle_complex $use_seq_cycle_complex \
     --fusion_type $fusion_type \
     --qkv $qkv
done

qkv='sfc'
for pred_len in 12 24 48 96
do
 python -u run.py \
     --is_training 1 \
     --root_path $root_path_name \
     --data_path $data_path_name \
     --model_id ${model_id_name}_${seq_len}_${pred_len}_sfc \
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
     --random_seed $random_seed \
     --lradj $lradj \
     --use_seq_cycle_complex $use_seq_cycle_complex \
     --fusion_type $fusion_type \
     --qkv $qkv
done

qkv='scf'
for pred_len in 12 24 48 96
do
 python -u run.py \
     --is_training 1 \
     --root_path $root_path_name \
     --data_path $data_path_name \
     --model_id ${model_id_name}_${seq_len}_${pred_len}_scf \
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
     --random_seed $random_seed \
     --lradj $lradj \
     --use_seq_cycle_complex $use_seq_cycle_complex \
     --fusion_type $fusion_type \
     --qkv $qkv
done

qkv='cfs'

fusion_type='time_add'
for pred_len in 12 24 48 96
do
  python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len}_time_add \
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
      --random_seed $random_seed \
      --lradj $lradj \
      --use_seq_cycle_complex $use_seq_cycle_complex \
      --fusion_type $fusion_type \
      --qkv $qkv
done

fusion_type='time_concat'
for pred_len in 12 24 48 96
do
  python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len}_time_concat \
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
      --random_seed $random_seed \
      --lradj $lradj \
      --use_seq_cycle_complex $use_seq_cycle_complex \
      --fusion_type $fusion_type \
      --qkv $qkv
done

fusion_type='freq'

use_seq_cycle_complex='cycle'
for pred_len in 12 24 48 96
do
  python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len}_cycle \
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
      --random_seed $random_seed \
      --lradj $lradj \
      --use_seq_cycle_complex $use_seq_cycle_complex \
      --fusion_type $fusion_type \
      --qkv $qkv
done

use_seq_cycle_complex='seq'
for pred_len in 12 24 48 96
do
  python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len}_seq \
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
      --random_seed $random_seed \
      --lradj $lradj \
      --use_seq_cycle_complex $use_seq_cycle_complex \
      --fusion_type $fusion_type \
      --qkv $qkv
done
