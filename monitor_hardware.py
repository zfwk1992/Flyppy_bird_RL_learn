#!/usr/bin/env python3
"""
硬件利用率实时监控脚本
监控CPU、GPU、内存使用情况，帮助诊断性能瓶颈
"""

import time
import psutil
import subprocess
import json
import threading
from datetime import datetime
import sys
import os

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  GPUtil未安装，将使用nvidia-smi监控GPU")

class HardwareMonitor:
    def __init__(self):
        self.monitoring = False
        self.log_file = None
        self.start_time = None
        
    def get_cpu_info(self):
        """获取CPU使用信息"""
        try:
            # CPU总体使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 各核心使用率
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            
            # 负载平均值
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            return {
                'total_percent': cpu_percent,
                'per_core': cpu_per_core,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2],
                'core_count': psutil.cpu_count(),
                'logical_count': psutil.cpu_count(logical=True)
            }
        except Exception as e:
            print(f"❌ CPU信息获取失败: {e}")
            return None
    
    def get_memory_info(self):
        """获取内存使用信息"""
        try:
            # 系统内存
            memory = psutil.virtual_memory()
            
            # 交换内存
            swap = psutil.swap_memory()
            
            return {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent
            }
        except Exception as e:
            print(f"❌ 内存信息获取失败: {e}")
            return None
    
    def get_gpu_info_gputil(self):
        """使用GPUtil获取GPU信息"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
                
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'temperature': gpu.temperature,
                    'utilization': gpu.load * 100,  # 转换为百分比
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_free_mb': gpu.memoryFree,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                })
            return gpu_info
        except Exception as e:
            print(f"❌ GPUtil获取GPU信息失败: {e}")
            return None
    
    def get_gpu_info_nvidia_smi(self):
        """使用nvidia-smi获取GPU信息"""
        try:
            # 执行nvidia-smi命令
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
                
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 7:
                        gpu_info.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'temperature': float(parts[2]) if parts[2] != '[N/A]' else 0,
                            'utilization': float(parts[3]) if parts[3] != '[N/A]' else 0,
                            'memory_total_mb': float(parts[4]),
                            'memory_used_mb': float(parts[5]),
                            'memory_free_mb': float(parts[6]),
                            'memory_percent': (float(parts[5]) / float(parts[4])) * 100
                        })
            
            return gpu_info if gpu_info else None
            
        except subprocess.TimeoutExpired:
            print("❌ nvidia-smi命令超时")
            return None
        except Exception as e:
            print(f"❌ nvidia-smi获取GPU信息失败: {e}")
            return None
    
    def get_gpu_info(self):
        """获取GPU信息 (尝试多种方法)"""
        # 优先使用GPUtil
        if GPU_AVAILABLE:
            gpu_info = self.get_gpu_info_gputil()
            if gpu_info:
                return gpu_info
        
        # 备用nvidia-smi
        return self.get_gpu_info_nvidia_smi()
    
    def get_process_info(self, process_name=None):
        """获取特定进程信息"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info']):
                try:
                    info = proc.info
                    if process_name and process_name.lower() not in info['name'].lower():
                        continue
                        
                    if info['cpu_percent'] > 1 or info['memory_percent'] > 1:  # 只显示占用较高的进程
                        processes.append({
                            'pid': info['pid'],
                            'name': info['name'],
                            'cpu_percent': info['cpu_percent'],
                            'memory_percent': info['memory_percent'],
                            'memory_mb': info['memory_info'].rss / (1024*1024) if info['memory_info'] else 0
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 按CPU使用率排序
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:10]  # 返回前10个
            
        except Exception as e:
            print(f"❌ 进程信息获取失败: {e}")
            return []
    
    def print_hardware_status(self):
        """打印硬件状态"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"🕐 硬件监控 - {timestamp}")
        print(f"{'='*60}")
        
        # CPU信息
        cpu_info = self.get_cpu_info()
        if cpu_info:
            print(f"🖥️  CPU状态:")
            print(f"   总体使用率: {cpu_info['total_percent']:.1f}%")
            print(f"   物理核心: {cpu_info['core_count']} | 逻辑核心: {cpu_info['logical_count']}")
            print(f"   当前频率: {cpu_info['frequency']:.0f} MHz")
            print(f"   负载平均: {cpu_info['load_avg_1m']:.2f} (1m) | {cpu_info['load_avg_5m']:.2f} (5m)")
            
            # 显示各核心使用率 (如果核心数不太多)
            if len(cpu_info['per_core']) <= 16:
                core_usage = " | ".join([f"C{i}:{usage:.0f}%" for i, usage in enumerate(cpu_info['per_core'])])
                print(f"   各核心: {core_usage}")
        
        # 内存信息
        memory_info = self.get_memory_info()
        if memory_info:
            print(f"\n💾 内存状态:")
            print(f"   系统内存: {memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB ({memory_info['percent']:.1f}%)")
            print(f"   可用内存: {memory_info['available_gb']:.1f}GB")
            if memory_info['swap_total_gb'] > 0:
                print(f"   交换内存: {memory_info['swap_used_gb']:.1f}GB / {memory_info['swap_total_gb']:.1f}GB ({memory_info['swap_percent']:.1f}%)")
        
        # GPU信息
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print(f"\n🎮 GPU状态:")
            for gpu in gpu_info:
                print(f"   GPU{gpu['id']}: {gpu['name']}")
                print(f"   利用率: {gpu['utilization']:.1f}% | 温度: {gpu['temperature']:.0f}°C")
                print(f"   显存: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB ({gpu['memory_percent']:.1f}%)")
        else:
            print(f"\n🎮 GPU状态: 未检测到NVIDIA GPU或驱动问题")
        
        # 高占用进程
        processes = self.get_process_info()
        if processes:
            print(f"\n🔥 高占用进程 (前5个):")
            for proc in processes[:5]:
                print(f"   {proc['name']} (PID:{proc['pid']}): CPU {proc['cpu_percent']:.1f}% | 内存 {proc['memory_mb']:.0f}MB")
        
        # Python相关进程
        python_processes = self.get_process_info('python')
        if python_processes:
            print(f"\n🐍 Python进程:")
            for proc in python_processes:
                print(f"   {proc['name']} (PID:{proc['pid']}): CPU {proc['cpu_percent']:.1f}% | 内存 {proc['memory_mb']:.0f}MB")
    
    def start_monitoring(self, interval=5, duration=None, log_to_file=True):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
        
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"hardware_monitor_{timestamp}.log"
            print(f"📝 监控日志将保存到: {self.log_file}")
        
        print(f"🚀 开始硬件监控 (间隔: {interval}秒)")
        if duration:
            print(f"   监控时长: {duration}秒")
        print("   按 Ctrl+C 停止监控")
        
        try:
            count = 0
            while self.monitoring:
                if duration and (time.time() - self.start_time) > duration:
                    break
                    
                # 清屏 (在终端中)
                if count > 0:
                    os.system('clear' if os.name == 'posix' else 'cls')
                
                self.print_hardware_status()
                
                # 记录到文件
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n=== {datetime.now().isoformat()} ===\n")
                        # 这里可以写入结构化数据，暂时简化
                
                count += 1
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  监控已停止")
        finally:
            self.monitoring = False
    
    def quick_check(self):
        """快速检查当前状态"""
        print("🔍 快速硬件状态检查")
        self.print_hardware_status()
        
        # 诊断建议
        print(f"\n💡 诊断建议:")
        
        cpu_info = self.get_cpu_info()
        gpu_info = self.get_gpu_info()
        memory_info = self.get_memory_info()
        
        if cpu_info and cpu_info['total_percent'] < 20:
            print("   🟡 CPU利用率很低 (<20%) - 可能存在I/O等待或进程阻塞")
        elif cpu_info and cpu_info['total_percent'] > 80:
            print("   🔴 CPU利用率很高 (>80%) - CPU可能是瓶颈")
        else:
            print("   🟢 CPU利用率正常")
        
        if gpu_info:
            gpu_util = gpu_info[0]['utilization']
            if gpu_util < 20:
                print("   🟡 GPU利用率很低 (<20%) - 可能CPU是瓶颈或批次太小")
            elif gpu_util > 90:
                print("   🔴 GPU利用率很高 (>90%) - GPU可能是瓶颈")
            else:
                print("   🟢 GPU利用率正常")
        
        if memory_info and memory_info['percent'] > 85:
            print("   🔴 内存使用率很高 (>85%) - 可能需要优化内存使用")
        else:
            print("   🟢 内存使用正常")

def main():
    """主函数"""
    monitor = HardwareMonitor()
    
    if len(sys.argv) == 1:
        # 默认: 快速检查
        monitor.quick_check()
    
    elif sys.argv[1] == 'monitor':
        # 持续监控模式
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else None
        monitor.start_monitoring(interval=interval, duration=duration)
    
    elif sys.argv[1] == 'check':
        # 快速检查模式
        monitor.quick_check()
    
    elif sys.argv[1] == 'help':
        print("""
🛠️  硬件监控脚本使用说明:

基本用法:
  python monitor_hardware.py                 # 快速检查
  python monitor_hardware.py check          # 快速检查  
  python monitor_hardware.py monitor        # 持续监控 (5秒间隔)
  python monitor_hardware.py monitor 3      # 持续监控 (3秒间隔)
  python monitor_hardware.py monitor 2 60   # 监控60秒 (2秒间隔)

监控指标:
  - CPU: 总体使用率、各核心使用率、频率、负载
  - GPU: 利用率、温度、显存使用 (NVIDIA)
  - 内存: 系统内存、交换内存使用情况
  - 进程: 高CPU/内存占用的进程列表

诊断建议:
  - CPU/GPU利用率低: 可能存在瓶颈或配置问题
  - CPU/GPU利用率高: 可能是性能瓶颈
  - 内存使用高: 可能需要优化内存配置
        """)
    
    else:
        print("❌ 未知参数，使用 'python monitor_hardware.py help' 查看帮助")

if __name__ == "__main__":
    main()