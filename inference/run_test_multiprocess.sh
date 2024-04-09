gpu_groups=(0 1 2 3 4 5 6 7)
N=${#gpu_groups[@]}
file=$1
model_path=$2
mode=$3 # pred or merge
base_name=`basename ${file}`
suffix=pred_spin_iterx # 预测文件的后缀，并行预测结束后，需再次合并，合并时，只合并包含该suffix的文件
if [[ $mode == "pred" ]]; then
    python split_excel_rows.py $file $N

    for idx in "${!gpu_groups[@]}"
    do
        export CUDA_VISIBLE_DEVICES=${gpu_groups[$idx]}
        nohup python test_chixu_shengcheng.py --model_path ${model_path} --file tmp/${base_name%.*}_sub${idx}.xlsx --input_output True --suffix ${suffix} > ./log/test${idx}_mrg.log 2>&1 &
    done
    # echo 'here'
else
    python merge_excel_rows.py tmp ${base_name%.*}_spin_iterx.xlsx ${suffix}
    rm -f tmp/*
fi