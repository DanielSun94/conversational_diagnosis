device=$1
model_name=$2
learning_rate=$3
value_net_length=$4

for repeat in 1 2 3 4 5
do
    echo "Repeat Num $repeat"
    for max_step in 10 20 40
    do
      echo "max step $max_step"
          python train.py --device=$device --model_name=$model_name --episode_max_len=$max_step --learning_rate=$learning_rate --value_net_length=$value_net_length
    done
done