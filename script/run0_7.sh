[ ${#1} -lt 1 ] && echo "error\n{1} datasets need to be set" && exit
[ ${#2} -lt 1 ] && echo "error\n{2} random seed need to be set" && exit
[ ${#3} -lt 1 ] && echo "error\n{3} query time need to be set" && exit
command="python run_baseline.py --datasets $1 --seed $2 --qt $3"
bias_v=0
for i in $(seq 0 7)
do
echo $i 
export CUDA_VISIBLE_DEVICES=$i 
echo $command --query_mode $((i+bias_v)) 
[ $i -lt "7" ] && $command --query_mode $((i+bias_v)) &
[ $i -eq "7" ] && $command --query_mode $((i+bias_v)) 

done
