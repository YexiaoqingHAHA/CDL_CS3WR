#! /bin/bash
result_path=./results
data_path=./data
for data_name in cf-a cf-t
do
  for ((k=5;k<=100;k=k+5))
  do
    for ((i=1;i<=20;i=i+1))
    do
    {
      ./ctr --directory $result_path/$data_name-$k-$i --user $data_path/$data_name/$data_name-train-$i-users.dat \
            --item $data_path/$data_name/$data_name-train-$i-items.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 \
            --num_factors $k --save_lag 20 --theta_opt

    }&
    done
    wait
  done
  wait
done

