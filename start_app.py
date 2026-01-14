"""
陶片聚类可视化应用启动脚本
"""

import argparse
import os
import sys
from pathlib import Path


def setup_environment():
    """设置环境变量"""
    # 添加当前目录到Python路径
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='陶片聚类交互可视化应用'
    )
    
    parser.add_argument('--port', type=int, default=9000,
                       help='Web服务端口 (默认: 9000)')
    
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')
    
    return parser.parse_args()


def apply_configuration(args):
    """应用配置到环境变量"""
    os.environ['CERAMIC_PORT'] = str(args.port)
    os.environ['CERAMIC_DEBUG'] = str(args.debug).lower()


def check_dependencies():
    """检查依赖项"""
    required_modules = [
        'dash', 'pandas', 'numpy', 'plotly', 
        'sklearn', 'PIL', 'umap'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ 缺少必要依赖项:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\n请安装缺少的依赖项:")
        print(f"   pip install {' '.join(missing_modules)}")
        return False
    
    return True


def print_startup_info(args):
    """打印启动信息"""
    print("🚀 陶片聚类可视化应用")
    print("=" * 40)
    print(f"🌐 服务地址: http://127.0.0.1:{args.port}")
    print(f"🔧 调试模式: {'开启' if args.debug else '关闭'}")
    print("=" * 40)


def main():
    """主函数"""
    # 设置环境
    setup_environment()
    
    # 解析参数
    args = parse_arguments()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 应用配置
    apply_configuration(args)
    
    # 打印启动信息
    print_startup_info(args)
    
    # 启动应用
    try:
        from app_clusters import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()