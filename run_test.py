#!/usr/bin/env python3
"""
Flappy Bird Dueling DQN 推理测试启动器
自动查找最新模型并调用 test/test_ai_gameplay.py 进行推理评测
"""
import os
import sys
import subprocess
import argparse

def find_latest_model(saved_networks_dir):
    """查找saved_networks目录下最新的.pth模型文件"""
    if not os.path.exists(saved_networks_dir):
        return None
    model_files = [f for f in os.listdir(saved_networks_dir) if f.endswith('.pth')]
    #model_files = [f for f in os.listdir(saved_networks_dir) if f.endswith('stable_best_bs128_378.pth')]
    if not model_files:
        return None
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(saved_networks_dir, x)), reverse=True)
    return os.path.join(saved_networks_dir, model_files[0])

def main():
    print("🎮 Flappy Bird Dueling DQN 推理测试启动器")
    print("=" * 50)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📂 当前工作目录: {os.getcwd()}")

    # 检查必要文件
    required_files = [
        "test/test_ai_gameplay.py",
        "saved_networks"
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 请确保在正确的项目目录中运行此脚本")
        return 1

    # 查找最新模型
    saved_networks_dir = "saved_networks"
    latest_model = find_latest_model(saved_networks_dir)
    if not latest_model:
        print("❌ 未找到任何.pth模型文件，请先训练模型！")
        return 1
    print(f"✅ 将加载最新模型: {latest_model}")

    # 解析可选参数并传递给测试脚本
    parser = argparse.ArgumentParser(description="Flappy Bird Dueling DQN 推理测试启动器")
    parser.add_argument('--episodes', type=int, help='测试局数')
    parser.add_argument('--fps', type=int, help='游戏帧率')
    parser.add_argument('--delay', type=float, help='每帧延迟（秒）')
    parser.add_argument('--no-q-values', action='store_true', help='不显示Q值')
    args, unknown = parser.parse_known_args()

    # 构建命令
    cmd = [
        sys.executable,
        "test/test_ai_gameplay.py",
        "--model", latest_model
    ]
    if args.episodes:
        cmd += ["--episodes", str(args.episodes)]
    if args.fps:
        cmd += ["--fps", str(args.fps)]
    if args.delay:
        cmd += ["--delay", str(args.delay)]
    if args.no_q_values:
        cmd.append("--no-q-values")
    # 传递未知参数（兼容性）
    cmd += unknown

    print("\n🚀 启动AI推理测试...")
    try:
        result = subprocess.run(cmd, cwd=script_dir)
        return result.returncode
    except Exception as e:
        print(f"❌ 启动测试失败: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        input("\n按 Enter 键退出...")
    sys.exit(exit_code)