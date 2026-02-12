#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计JSON文件中input_file字段包含的文件数量
文件数量 = 换行符数量 + 1
按原始数据定义（.jsonl文件）统计数据条数
"""

import json
import os
from pathlib import Path
from collections import defaultdict


def count_files_in_input_file(input_file_str):
    """
    统计input_file字段中的文件数量
    文件数量 = 换行符数量 + 1
    """
    if not input_file_str:
        return 0
    return input_file_str.count('\n') + 1


def load_original_data(jsonl_path):
    """
    从原始 .jsonl 文件中加载所有数据
    返回 {id: input_file} 的字典
    """
    data_dict = {}
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                data_dict[data['id']] = data.get('input_file', '')
    except Exception as e:
        print(f"❌ 读取原始数据文件 {jsonl_path} 时出错: {e}")
    return data_dict


def analyze_directory(directory_path, original_jsonl_path):
    """
    分析指定目录下所有JSON文件的input_file字段
    基于原始数据定义统计
    返回统计结果
    """
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        print(f"⚠️  目录不存在: {directory_path}")
        return None
    
    # 从原始 jsonl 文件加载所有数据
    original_data = load_original_data(original_jsonl_path)
    if not original_data:
        print(f"⚠️  无法加载原始数据定义: {original_jsonl_path}")
        return None
    
    original_total = len(original_data)
    
    results = []
    total_file_count = 0
    missing_items = []
    
    for data_id, original_input_file in original_data.items():
        json_file_path = dir_path / f"{data_id}.json"
        
        if not json_file_path.exists():
            # 缺失评估结果，但仍统计原始定义的 input_file
            file_count = count_files_in_input_file(original_input_file)
            missing_items.append({
                'id': data_id,
                'input_file': original_input_file,
                'file_count': file_count
            })
            
            # 统计到结果中（标记为缺失）
            results.append({
                'json_file': f"{data_id}.json",
                'input_file': original_input_file,
                'file_count': file_count,
                'is_missing': True
            })
            
            total_file_count += file_count
            continue
            
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            input_file = data.get('input_file', '')
            file_count = count_files_in_input_file(input_file)
            
            results.append({
                'json_file': json_file_path.name,
                'input_file': input_file,
                'file_count': file_count,
                'is_missing': False
            })
            
            total_file_count += file_count
            
        except Exception as e:
            print(f"❌ 处理文件 {json_file_path.name} 时出错: {e}")
            # 出错时使用原始定义
            file_count = count_files_in_input_file(original_input_file)
            missing_items.append({
                'id': data_id,
                'input_file': original_input_file,
                'file_count': file_count
            })
            results.append({
                'json_file': f"{data_id}.json",
                'input_file': original_input_file,
                'file_count': file_count,
                'is_missing': True
            })
            total_file_count += file_count
    
    return {
        'directory': directory_path,
        'original_total': original_total,
        'total_json_files': len([r for r in results if not r['is_missing']]),
        'total_input_files': total_file_count,
        'missing_count': len(missing_items),
        'missing_items': missing_items,
        'details': results
    }


def count_by_file_number(results):
    """
    按input_file数量分组统计
    返回：0个文件、1个文件、2个文件、>=3个文件的数据条数
    """
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3_plus = 0
    
    for item in results:
        file_count = item['file_count']
        if file_count == 0:
            count_0 += 1
        elif file_count == 1:
            count_1 += 1
        elif file_count == 2:
            count_2 += 1
        elif file_count >= 3:
            count_3_plus += 1
    
    return {
        'count_0_file': count_0,
        'count_1_file': count_1,
        'count_2_files': count_2,
        'count_3plus_files': count_3_plus
    }


def main():
    # 定义要分析的目录和对应的原始数据文件
    configs = [
        {
            'name': 'chart',
            'eval_dir': 'output/evals/deepseek_reasoner/chart',
            'jsonl': 'data/chart/chart.jsonl'
        },
        {
            'name': 'file_generation',
            'eval_dir': 'output/evals/deepseek_reasoner/file_generation',
            'jsonl': 'data/file_generation/file_generation.jsonl'
        },
        {
            'name': 'numeric',
            'eval_dir': 'output/evals/deepseek_reasoner/numeric',
            'jsonl': 'data/numeric/numeric.jsonl'
        }
    ]
    
    print("=" * 80)
    print("统计 input_file 字段包含的文件数量")
    print("（基于原始数据定义统计）")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for config in configs:
        directory = config['eval_dir']
        jsonl_path = config['jsonl']
        
        print(f"\n📁 分析目录: {directory}")
        print(f"📄 原始数据: {jsonl_path}")
        print("-" * 80)
        
        result = analyze_directory(directory, jsonl_path)
        
        if result:
            all_results[config['name']] = result
            
            print(f"📊 原始数据定义: {result['original_total']} 条")
            print(f"✅ 有评估结果: {result['total_json_files']} 条")
            if result['missing_count'] > 0:
                print(f"⚠️  缺失评估结果: {result['missing_count']} 条")
                for item in result['missing_items']:
                    print(f"   - {item['id']}: {item['file_count']} 个文件 ({item['input_file'][:50]}...)" if len(item['input_file']) > 50 else f"   - {item['id']}: {item['file_count']} 个文件 ({item['input_file']})")
            
            print(f"✅ input_file中的文件总数: {result['total_input_files']}")
            if result['total_json_files'] > 0:
                print(f"✅ 平均每个JSON包含文件数: {result['total_input_files'] / result['total_json_files']:.2f}")
            
            # 按文件数量分组统计
            group_stats = count_by_file_number(result['details'])
            print(f"\n📈 按input_file数量分组:")
            if group_stats['count_0_file'] > 0:
                print(f"  - 0个文件的数据条数: {group_stats['count_0_file']}")
            print(f"  - 1个文件的数据条数: {group_stats['count_1_file']}")
            print(f"  - 2个文件的数据条数: {group_stats['count_2_files']}")
            print(f"  - ≥3个文件的数据条数: {group_stats['count_3plus_files']}")
            
            # 验证总数
            total_counted = group_stats['count_0_file'] + group_stats['count_1_file'] + group_stats['count_2_files'] + group_stats['count_3plus_files']
            print(f"  ✓ 验证: {group_stats['count_0_file']} + {group_stats['count_1_file']} + {group_stats['count_2_files']} + {group_stats['count_3plus_files']} = {total_counted}")
            
            # 打印0个文件的详细信息
            if group_stats['count_0_file'] > 0:
                print(f"\n⚠️  0个文件的详细信息:")
                for item in result['details']:
                    if item['file_count'] == 0:
                        print(f"  - {item['json_file']}: input_file = '{item['input_file']}'")
            
            # 将分组统计添加到结果中
            result['group_statistics'] = group_stats
            
            # 显示一些示例
            print(f"\n前5个文件的详情:")
            for item in result['details'][:5]:
                print(f"  - {item['json_file']}: {item['file_count']} 个文件")
                if item['file_count'] > 1:
                    files = item['input_file'].split('\n')
                    for i, f in enumerate(files, 1):
                        print(f"      {i}. {f}")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 汇总统计")
    print("=" * 80)
    
    for name, result in all_results.items():
        print(f"\n【{name.upper()}】")
        print(f"  原始数据定义: {result['original_total']} 条")
        print(f"  有评估结果: {result['total_json_files']} 条")
        if result['missing_count'] > 0:
            print(f"  缺失评估结果: {result['missing_count']} 条")
        print(f"  input_file总文件数: {result['total_input_files']}")
        if result['total_json_files'] > 0:
            print(f"  平均文件数: {result['total_input_files'] / result['total_json_files']:.2f}")
        
        group_stats = result['group_statistics']
        if group_stats['count_0_file'] > 0:
            print(f"  ├─ 0个文件: {group_stats['count_0_file']} 条")
        print(f"  ├─ 1个文件: {group_stats['count_1_file']} 条")
        print(f"  ├─ 2个文件: {group_stats['count_2_files']} 条")
        print(f"  └─ ≥3个文件: {group_stats['count_3plus_files']} 条")
    
    # 按文件数量分组的汇总表格
    print("\n" + "=" * 80)
    print("📋 分组统计表格（按原始数据定义）")
    print("=" * 80)
    print(f"\n{'类型':<20} {'原始定义':<12} {'0个文件':<12} {'1个文件':<12} {'2个文件':<12} {'≥3个文件':<12} {'缺失':<12}")
    print("-" * 100)
    
    for name, result in all_results.items():
        group_stats = result['group_statistics']
        original = result['original_total']
        missing = result['missing_count']
        
        print(f"{name:<20} {original:<12} {group_stats['count_0_file']:<12} {group_stats['count_1_file']:<12} {group_stats['count_2_files']:<12} {group_stats['count_3plus_files']:<12} {missing:<12}")
    
    # 保存详细结果到JSON文件
    output_file = 'input_file_statistics.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
