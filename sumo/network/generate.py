"""
SUMO 路网生成脚本

两步：
1. netgenerate --grid 生成网格路网
2. netconvert 添加交通灯
"""

import os
import subprocess
import sys
from pathlib import Path

SUMO_DIR = Path(__file__).parent

SUMO_HOME = os.environ.get("SUMO_HOME", "")
if SUMO_HOME:
    NETGENERATE = os.path.join(SUMO_HOME, "bin", "netgenerate.exe")
    NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert.exe")
else:
    NETGENERATE = "netgenerate"
    NETCONVERT = "netconvert"


def generate_network():
    """生成带交通灯的十字路口路网"""
    net_file = SUMO_DIR / "intersection.net.xml"
    tmp_file = SUMO_DIR / "_tmp.net.xml"

    # Step 1: netgenerate 生成基础网格
    cmd = [
        NETGENERATE,
        "--grid",
        "--grid.number", "1",
        "--grid.length", "200",
        "--grid.attach-length", "100",
        "--output-file", str(tmp_file),
        "--no-turnarounds",
    ]
    print(f"Step 1: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        sys.exit(1)

    # Step 2: netconvert 添加交通灯
    cmd = [
        NETCONVERT,
        "--sumo-net-file", str(tmp_file),
        "--output-file", str(net_file),
        "--tls.set", "A0",
        "--tls.guess",
    ]
    print(f"Step 2: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        sys.exit(1)

    tmp_file.unlink(missing_ok=True)
    print(f"路网已生成: {net_file}")
    return net_file


if __name__ == "__main__":
    try:
        generate_network()
        print("\n生成完成！")
    except FileNotFoundError:
        print("错误: 未找到 netgenerate")
        sys.exit(1)
