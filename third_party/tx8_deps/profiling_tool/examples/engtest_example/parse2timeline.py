################################################################################
##  脚本作用:将采集到的原始指令性能数据解析成timeline json格式文件,           ##
##         进而可以在perfetto、chrome/tracing等工具中可视化。                 ##
##         支持连续传入多个原始指令性能数据文件, 解析完成后会在               ##
##         第一个原始指令性能数据文件同路径下生成解析后的json文件。           ##
##  运行方式: python3 parse2timeline2.py group_time.csv ...                   ##
##  版本：V1.0                                                                ##
##  原始指令性能数据格式：                                                    ##
##         device_id,tile_id,group_id,instr_type,start_time,duration          ##
##         示例：0,6,0,RDMA,3049813569,355393                                 ##
################################################################################

import os
import sys
import json

def parse_instr_csv_to_events(input_files):
    # 一个tile对应一个pid
    pid = 0
    # 不同类型指令对应不同tid，用于在不同行展示
    tid_map = {
        'RDMA': 0,
        'NE': 1,
        'WDMA': 2,
        'TDMA': 3,
        'CT': 4,
        'DTE': 5,
        'NOCGEMM': 6, #预留
    }

    # TX81时间Timer寄存器是32位,能表示的最大时间值是UINT32，超过后会反转
    # 所以处理start_time时要根据反转次数进行修正
    UINT32_MAX_VAL = 2**32-1

    # 存储不同类型的事件（使用集合来去重）
    event_sets = {key: set() for key in tid_map.keys()}

   # 存储每个pid_tid事件的个数用于每个事件的name
    pid_tid_events_count_map = dict()

    # 用于记录所有事件的最早开始时间，作为全局基准
    global_base_time = None

    for file in input_files:
        try:
            with open(file, "r") as f:
                ori_data = f.readlines()
        except IOError as e:
            print(f"无法读取文件 {file}: {e}")
            continue

        print(f"{file} 行数: {len(ori_data)}")

        REGISTER_INVERT_COUNT = 0
        last_start_time = 0
        for line_num, line in enumerate(ori_data, 1):
            line = line.strip()
            if not line:
                continue

            vect = line.split(',')
            # 校验数据格式是否正确（至少6个字段）
            if len(vect) < 6 or line_num == 1:
                #print(f"警告: 第 {line_num} 行格式错误(字段不足6个)，跳过: {line}")
                continue

            # 尝试从多个位置提取指令类型，增加容错性
            possible_types = [vect[3].strip(), vect[0].strip(), vect[1].strip()]
            cmd_type = None
            for typ in possible_types:
                if typ in tid_map:
                    cmd_type = typ
                    break

            if cmd_type is None:
                print(f"警告: 第 {line_num} 行未找到已知指令类型，内容: {line}")
                continue

            try:
                # 转换时间（纳秒 -> 微秒）
                # 尝试不同的时间字段索引，应对可能的格式差异
                start_idx, dur_idx = 4, 5
                if len(vect) > 6:  # 处理可能的额外字段情况
                    start_idx, dur_idx = 5, 6

                if float(vect[start_idx]) < last_start_time and (last_start_time-float(vect[start_idx]))/1000/1000/1000 > 1.0:
                    #时间寄存器出现反转
                    REGISTER_INVERT_COUNT += 1
                    print(f"警告: 从第 {line_num} 行开始start time出现反转，内容: {line}")

                last_start_time = float(vect[start_idx])
                start_time = (float(vect[start_idx]) + REGISTER_INVERT_COUNT*UINT32_MAX_VAL) / 1000.0
                duration = float(vect[dur_idx]) / 1000.0
            except (ValueError, IndexError) as e:
                print(f"警告: 第 {line_num} 行时间解析错误 '{e}'，内容: {line}")
                continue

            # 计算全局基准时间（所有文件中最早的开始时间）
            if global_base_time is None or start_time < global_base_time:
                global_base_time = start_time

            #统计pid tid(即cmd_type)下指令数量
            if vect[1]+cmd_type in pid_tid_events_count_map:
                pid_tid_events_count_map[vect[1]+cmd_type] += 1
            else:
                pid_tid_events_count_map[vect[1]+cmd_type] = 1

            # 创建事件对象（转换为可哈希的元组用于去重）
            event_tuple = (
                'X',  # ph
                cmd_type,  # cat
                f"group{vect[2]}_{cmd_type}{str(pid_tid_events_count_map[vect[1]+cmd_type])}",  # name
                int(vect[1]),  # pid
                tid_map[cmd_type],  # tid
                start_time,  # ts（原始时间）
                duration  # dur
            )
            if event_tuple in event_sets[cmd_type]:
                print(f"警告: 第 {line_num} 行解析出的event_tuple已存在，重复，内容: {line}")
            else:
                event_sets[cmd_type].add(event_tuple)

    # 转换回列表并保留原始结构
    event_lists = {}
    for cmd_type in event_sets:
        # 将元组转换回字典
        event_lists[cmd_type] = [
            {
                'ph': t[0],
                'cat': t[1],
                'name': t[2],
                'pid': t[3],
                'tid': t[4],
                'ts': t[5],
                'dur': t[6]
            } for t in event_sets[cmd_type]
        ]

    # 如果没有有效事件，直接退出
    if global_base_time is None:
        print("错误: 未找到任何有效事件数据")
        sys.exit(1)

    print('global_base_time:', global_base_time)
    # 统一调整所有事件的时间，以全局最早时间为基准
    all_events = []
    for cmd_type in event_lists:
        for event in event_lists[cmd_type]:
            event['ts'] -= global_base_time  # 调整为相对时间
            all_events.append(event)

    all_events.sort(key=lambda x: (x['pid'], x['tid'], x['cat'], x['name']))

    # 输出各类型事件数量
    for cmd_type in tid_map:
        print(f'{cmd_type.ljust(5)}数量: {len(event_lists[cmd_type])}')
    total = sum(len(lst) for lst in event_lists.values())
    print(f'Total数量: {total}')
    return all_events

def parse_group_csv_to_events(file):
    # 一个tile对应一个pid
    pid = 0

    # TX81时间Timer寄存器是32位,能表示的最大时间值是UINT32，超过后会反转
    # 所以处理start_time时要根据反转次数进行修正
    UINT32_MAX_VAL = 2**32-1

    # 用于记录所有事件的最早开始时间，作为全局基准
    global_base_time = None

    event_lists = []

    try:
        with open(file, "r") as f:
            ori_data = f.readlines()
    except IOError as e:
        print(f"无法读取文件 {file}: {e}")
        return []

    print(f"{file} 行数: {len(ori_data)}")

    REGISTER_INVERT_COUNT = 0
    last_start_time = 0
    for line_num, line in enumerate(ori_data, 1):
        line = line.strip()
        if not line:
            continue

        vect = line.split(',')
        # 校验数据格式是否正确（至少12个字段）
        if len(vect) < 12 or line_num == 1:
            #print(f"警告: 第 {line_num} 行格式错误(字段不足12个)，跳过: {line}")
            continue

        try:
            # 尝试不同的时间字段索引，应对可能的格式差异
            start_idx, dur_idx = 4, 5
            # if float(vect[start_idx]) < last_start_time:
            #     #时间寄存器出现反转
            #     REGISTER_INVERT_COUNT += 1
            #     print(f"警告: 从第 {line_num} 行开始start time出现反转，内容: {line}")

            last_start_time = float(vect[start_idx])
            start_time = (float(vect[start_idx]) + REGISTER_INVERT_COUNT*UINT32_MAX_VAL)
            duration = float(vect[dur_idx])
        except (ValueError, IndexError) as e:
            print(f"警告: 第 {line_num} 行时间解析错误 '{e}'，内容: {line}")
            continue

        # 计算全局基准时间（所有文件中最早的开始时间）
        if global_base_time is None or start_time < global_base_time:
            global_base_time = start_time

        # 创建事件对象
        event_tuple = {
            'ph': 'X',
            'cat': vect[2],
            'name': f"{vect[2]}_{vect[3]}",
            'pid': int(vect[1]),
            'tid': 0,
            'ts': start_time,
            'dur': duration
        }
        event_lists.append(event_tuple)

    # 如果没有有效事件，直接退出
    if global_base_time is None:
        print("错误: 未找到任何有效事件数据")
        sys.exit(1)

    print('global_base_time:', global_base_time)
    # 统一调整所有事件的时间，以全局最早时间为基准
    all_events = []
    for event in event_lists:
        event['ts'] -= global_base_time  # 调整为相对时间
        all_events.append(event)

    all_events.sort(key=lambda x: (x['pid'], x['tid'], x['cat'], x['name']))

    total = len(event_lists)
    print(f'Total数量: {total}')

    # 生成输出文件名（使用第一个输入文件的名称）
    if len(sys.argv) > 1:
        output_file = f"{os.path.splitext(sys.argv[1])[0]}.timeline.json"
        timeline = {'traceEvents': all_events}

    try:
        with open(output_file, "w") as f:
            json.dump(timeline, f)
        print(f"时间线文件已生成: {output_file}")
    except IOError as e:
        print(f"无法写入输出文件 {output_file}: {e}")

    return all_events

def merge_csv_to_timeline(input_file, input_files):
    all_events = []
    parse_group_csv_to_events(input_file)
    all_events.extend(parse_instr_csv_to_events(input_files))

    all_events.sort(key=lambda x: (x['pid'], x['tid'], x['cat'], x['name']))

    # 生成输出文件名（使用第一个输入文件的名称）
    if len(sys.argv) > 1:
        temp_output_file = f"{os.path.splitext(sys.argv[2])[0]}.timeline.json"
        output_file = temp_output_file.replace("ne_dte_instr_time_trace.timeline.json", "instr_time_trace.timeline.json")
        timeline = {'traceEvents': all_events}

    try:
        with open(output_file, "w") as f:
            json.dump(timeline, f)
        print(f"时间线文件已生成: {output_file}")
    except IOError as e:
        print(f"无法写入输出文件 {output_file}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 parseGroup2timeline.py <group_time_csv_path> <instr_time_trace_csv_path>")
        print("example: python3 parseGroup2timeline.py ./result/profilier_out/group_time.csv ./result/profilier_out/instr_time_trace.csv")
    input_file = sys.argv[1]
    input_files = sys.argv[2:]

    for file in input_files:
        if not file.endswith('.csv'):
            print(f"错误:输入文件{file}不是CSV格式")
            sys.exit(1)

    merge_csv_to_timeline(input_file, input_files)
