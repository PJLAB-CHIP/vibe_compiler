#!/bin/bash

# 定义要下载的文件及对应的本地文件名
files=(
    "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81fw_202602261758_b72af3.tar.gz tx81fw_202602261758_b72af3.tar.gz"
    "http://172.50.1.66:8082/artifactory/tx8-generic-dev/tx81fw/tx81score-profiling-fw_202602271353_48576e.tar.gz tx81score-profiling-fw_202602271353_48576e.tar.gz"
)

# 检查并下载文件（显式输出完整下载命令）
for file_info in "${files[@]}"; do
    download_url=$(echo "$file_info" | awk '{print $1}')
    local_file=$(echo "$file_info" | awk '{print $2}')
    if [ -f "$local_file" ]; then
        echo "本地已有 $local_file，跳过下载"
    else
        echo "执行下载命令：wget -qO \"$local_file\" \"$download_url\""  # 输出完整下载命令
        wget -qO "$local_file" "$download_url"
        if [ $? -ne 0 ]; then
            echo "下载失败！请手动执行以下命令重试："
            echo "wget -qO \"$local_file\" \"$download_url\""  # 失败时再次输出命令
            exit 1
        fi
    fi
done

# 解压文件并记录解压后的文件夹名称
unzipped_folders=()  # 存储解压后的文件夹名
for file_info in "${files[@]}"; do
    local_file=$(echo "$file_info" | awk '{print $2}')
    # 解压文件（假设tar.gz解压后生成与文件名前缀相同的文件夹）
    tar -zxvf "$local_file"
    if [ $? -ne 0 ]; then
        echo "解压 $local_file 失败"
        exit 1
    fi
    # 获取解压后的文件夹名（去掉.tar.gz后缀）
    folder_name="${local_file%.tar.gz}"
    unzipped_folders+=("$folder_name")  # 添加到数组中
done

fw_path=./tx81fw-rtt/bin/FW
card_num=$(lspci -n | grep '200c:0001' | wc -l)

for ((i=0; i<$card_num; i++)); do
    echo "off" > /sys/kernel/debug/accel/dev-$i/fw/all
done

if [ -d "/lib/firmware/accel/dynamic" ]; then
    cp $fw_path/kcore.tufw /lib/firmware/accel/dynamic/
    fw_path=./tx81score-profiling-fw
    cp $fw_path/score0.tufw /lib/firmware/accel/dynamic/
    cp $fw_path/score1.tufw /lib/firmware/accel/dynamic/
else
    cp $fw_path/kcore.tufw /lib/firmware/accel/
    fw_path=./tx81score-profiling-fw
    cp $fw_path/score0.tufw /lib/firmware/accel/
    cp $fw_path/score1.tufw /lib/firmware/accel/
fi

for ((i=0; i<$card_num; i++)); do
    echo "on" > /sys/kernel/debug/accel/dev-$i/fw/all
done

status_info=$(cat /sys/kernel/debug/accel/dev-0/fw/all)
kcore_status=${status_info#*kcore[}
kcore_status=${kcore_status%%]*}
score0_status=${status_info#*score0[}
score0_status=${score0_status%%]*}
score1_status=${status_info#*score1[}
score1_status=${score1_status%%]*}

if [ "$kcore_status" = "on" ] && [ "$score0_status" = "on" ] && [ "$score1_status" = "on" ]; then
    echo "execute success!"
    # 清理下载的tar.gz文件和解压后的文件夹
    for file_info in "${files[@]}"; do
        local_file=$(echo "$file_info" | awk '{print $2}')
        rm -f "$local_file"  # 删除下载的压缩包
    done
    for folder in "${unzipped_folders[@]}"; do
	echo "$folder"
        rm -rf "$folder"  # 删除解压后的文件夹
    done
    rm -rf tx81fw-rtt
    rm -rf "$fw_path"
else
    echo "execute failed! kcore: $kcore_status, score0: $score0_status, score1: $score1_status"
fi

