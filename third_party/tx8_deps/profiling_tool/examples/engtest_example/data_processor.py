
import csv
import os
import sys
from collections import defaultdict

class InstrTrace:
    def __init__(self, chip_id, tile_id, group_id, type_, start_time, duration):
        self.chip_id = int(chip_id)
        self.tile_id = int(tile_id)
        self.group_id = int(group_id)
        self.type = type_
        self.start_time = int(start_time)
        self.duration = int(duration)

class ParamInfo:
    def __init__(self, chip_id, tile_id, group_id, type_, total_length):
        self.chip_id = int(chip_id)
        self.tile_id = int(tile_id)
        self.group_id = int(group_id)
        self.type = type_
        self.total_length = int(total_length)

def read_instr_time_trace(filename):
    traces = []
    #print(f"开始读取指令时间跟踪文件: {filename}")
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # skip header
            #print(f"文件头: {header}")
            for i, row in enumerate(reader):
                if len(row) >= 6:
                    trace = InstrTrace(row[0], row[1], row[2], row[3], row[4], row[5])
                    traces.append(trace)
                    # if i < 5:  # 打印前5条记录用于调试
                        # print(f"读取到跟踪记录[{i}]: chip_id={trace.chip_id}, tile_id={trace.tile_id}, "
                        #      f"group_id={trace.group_id}, type={trace.type}")
                else:
                    print(f"警告: 第{i+2}行数据列数不足: {row}")
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 不存在")
        return traces
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return traces

    #print(f"总共读取到 {len(traces)} 条指令跟踪记录")
    return traces

def read_param_info(filename):
    param_list = []
    #print(f"开始读取参数信息文件: {filename}")
    if not os.path.exists(filename):
        print(f"警告: 参数文件 {filename} 不存在")
        return param_list

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader)  # skip header
            #print(f"参数文件头: {header}")
            for i, row in enumerate(reader):
                if len(row) >= 5:
                    info = ParamInfo(row[0], row[1], row[2], row[3], row[4])
                    param_list.append(info)
                    # if i < 5:  # 打印前5条记录用于调试
                        # print(f"读取到参数记录[{i}]: chip_id={info.chip_id}, tile_id={info.tile_id}, "
                        #      f"group_id={info.group_id}, type={info.type}, total_length={info.total_length}")
                else:
                    print(f"警告: 参数文件第{i+2}行数据列数不足: {row}")
    except Exception as e:
        print(f"读取参数文件时发生错误: {e}")
        return param_list

    #print(f"总共读取到 {len(param_list)} 条参数记录")
    return param_list

def create_param_index(param_list):
    """创建基于(chip_id, tile_id, group_id, type)的索引"""
    index = {}
    for info in param_list:
        key = (info.chip_id, info.tile_id, info.group_id, info.type)
        index[key] = info
    return index

def write_output(results, output_file):
    print(f"开始写入结果到文件: {output_file}")
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["chip_id", "tile_id", "group_id", "type", "total_length", "duration", "bandwidth(TFLOPS)", "utilization(%)"])
            for info, duration, source_type in results:
                bandwidth = info.total_length / duration / 1000 if duration > 0 else 0
                if source_type == "CT":
                    theoretical_flops = 0.032
                else:
                    theoretical_flops = 8.192
                utilization = min(bandwidth / theoretical_flops * 100, 100.0) 
                writer.writerow([
                    info.chip_id, info.tile_id, info.group_id,
                    info.type, info.total_length, duration,
                    f"{bandwidth:.6f}", f"{utilization:.2f}"
                ])
        print(f"成功写入 {len(results)} 条记录到输出文件")
    except Exception as e:
        print(f"写入输出文件时发生错误: {e}")

def process_param_data(traces, param_filename, source_type):
    """处理特定类型的参数数据"""
    # print(f"\n=== 处理{source_type}数据 ===")

    # 存储结果
    results = []
    match_success_count = 0
    match_fail_count = 0

    # 按tile_id分组处理，避免重复读取相同文件
    tile_groups = defaultdict(list)
    for trace in traces:
        tile_groups[trace.tile_id].append(trace)

    # 处理每个tile_id的数据
    for tile_id, tile_traces in tile_groups.items():
        # 读取对应的参数文件
        param_list = read_param_info(param_filename)

        if not param_list:
            print(f"跳过 tile_id={tile_id}，因为未读取到{source_type}参数数据")
            continue

        # 创建索引以便快速查找
        param_index = create_param_index(param_list)

        # 匹配并收集数据
        for trace in tile_traces:
            # 使用(chip_id, tile_id, group_id, type)作为匹配键
            key = (trace.chip_id, trace.tile_id, trace.group_id, trace.type)

            if key in param_index:
                results.append((param_index[key], trace.duration, source_type))
                match_success_count += 1
                # if match_success_count <= 5:  # 打印前5次成功的匹配用于调试
                    # print(f"成功匹配{source_type}: {key} -> duration={trace.duration}")
            else:
                match_fail_count += 1
                # if match_fail_count <= 10:  # 打印前10次失败的匹配用于调试
                    # print(f"匹配{source_type}失败: {key} 未在参数中找到对应记录")

    print(f"{source_type}匹配统计: 成功={match_success_count}, 失败={match_fail_count}")
    return results

def main():
    print("=== 开始数据处理 ===")
    args = sys.argv[1:]
    trace_file = args[0] if len(args) > 0 else "instr_time_trace.csv"
    ct_param_file = args[1] if len(args) > 1 else "ct_param_info.csv"
    ne_param_file = args[2] if len(args) > 2 else "ne_param_info.csv"
    output_file = args[3] if len(args) > 3 else "instr_time_perf.csv"

    print("=== 开始数据处理 ===")
    print(f"指令时间跟踪文件: {trace_file}")
    print(f"CT参数信息文件: {ct_param_file}")
    print(f"NE参数信息文件: {ne_param_file}")
    print(f"输出文件: {output_file}")

    # 读取指令时间跟踪数据
    traces = read_instr_time_trace(trace_file)

    if not traces:
        print("未能读取到任何指令跟踪数据，程序退出")
        return

    # 处理CT参数数据
    ct_results = process_param_data(traces, ct_param_file, "CT")

    # 处理NE参数数据
    ne_results = process_param_data(traces, ne_param_file, "NE")

    # 合并结果
    all_results = ct_results + ne_results
    print(f"\n=== 总体统计 ===")
    print(f"CT参数匹配结果: {len(ct_results)} 条记录")
    print(f"NE参数匹配结果: {len(ne_results)} 条记录")
    print(f"总计匹配结果: {len(all_results)} 条记录")

    if not all_results:
        print("警告: 未找到任何匹配的数据!")
        # 显示一些调试信息帮助排查问题
        if traces:
            print("\n第一条指令跟踪数据:")
            first_trace = traces[0]
            print(f"  chip_id: {first_trace.chip_id}")
            print(f"  tile_id: {first_trace.tile_id}")
            print(f"  group_id: {first_trace.group_id}")
            print(f"  type: '{first_trace.type}'")

        for param_file in [ct_param_file, ne_param_file]:
            if os.path.exists(param_file):
                print(f"\n尝试读取{param_file}的第一条记录:")
                try:
                    with open(param_file, 'r', encoding='utf-8') as f:
                        lines = list(csv.reader(f))
                        if len(lines) > 1:
                            header = lines[0]
                            first_data = lines[1]
                            print(f"  文件头: {header}")
                            print(f"  第一条数据: {first_data}")
                        else:
                            print("  文件为空或只有头部")
                except Exception as e:
                    print(f"  读取文件时出错: {e}")

    # 输出结果到CSV文件
    write_output(all_results, output_file)
    print(f"\n处理完成，共处理 {len(all_results)} 条记录，结果已保存到 instr_time_perf.csv")

if __name__ == "__main__":
    main()
