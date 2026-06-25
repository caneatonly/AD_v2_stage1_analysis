"""Free-decay 实验串口采集程序.

用途
----
从机体固件 (Acoustic_decoy_v2) 的 USART1 遥测口读取 IMU 数据,
按键控制录制一次 "激励 -> 自由衰减 -> 稳定" 过程, 并按照
sim_flip 识别管线要求的格式落盘到 ``sim_flip/data/raw/runs/``.

链路事实 (已对齐固件与管线)
---------------------------
* 固件 ``Task_Telemetry`` 每 20 ms (~50 Hz) 输出一行::

      angleX=%.2f,angleY=%.2f,angleZ=%.2f,gyroX=%.2f,gyroY=%.2f,gyroZ=%.2f,time=%lu\\r\\n

  经 ``console_printf -> vprintf -> __io_putchar -> HAL_UART_Transmit(&huart1)``
  发出, ``huart1`` 波特率 19200, 8N1.
* ``time = xTaskGetTickCount()``, ``configTICK_RATE_HZ = 1000``, 即毫秒 tick.
* 管线 ``raw_preprocess._load_raw_txt`` 用 ``sep=\\s+`` 读取, 需要列
  ``angleX angleY gyroX gyroY time`` (带表头, 空白分隔, 仅 5 列), 且 time 严格递增.

设计要点
--------
* 双线程: 读线程持续解析串口行并入队; 主线程处理按键与显示.
* 实时 θ 仅为操作辅助: 滚动缓冲做 PCA 投影 (θ_phys ≈ 90 + proj),
  与管线最终复算值不必完全一致; 同时并排显示原始 angleX/angleY 兜底.
* 落盘前校验: time 严格递增 (非递增样本静默丢弃并计数), 行数 >= 阈值.
* 主文件 5 列喂管线; 另存含 Z 轴的完整原始备份便于事后排查.

用法
----
    python collect_freedecay.py                # 启动后列出可用串口供选择
    python collect_freedecay.py --port COM5    # 直接指定串口

依赖: pyserial (已在 pytorchlearning 环境安装).
"""

from __future__ import annotations

import argparse
import csv
import queue
import sys
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - 环境兜底提示
    sys.stderr.write(
        "缺少 pyserial. 请在 sim_flip 环境安装:\n"
        "    python -m pip install pyserial\n"
    )
    raise

try:
    import msvcrt  # Windows 非阻塞按键
    _HAS_MSVCRT = True
except ImportError:  # pragma: no cover - 非 Windows 兜底
    _HAS_MSVCRT = False


# --------------------------------------------------------------------------- #
# 路径与常量
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "sim_flip" / "data" / "raw" / "runs"
BACKUP_DIR = SCRIPT_DIR / "raw_backup"

PIPELINE_COLS = ["angleX", "angleY", "gyroX", "gyroY", "time"]
FULL_COLS = ["angleX", "angleY", "angleZ", "gyroX", "gyroY", "gyroZ", "time"]

DEFAULT_BAUD = 19200
MIN_SAMPLES = 500          # 落盘前最小样本数 (~10 s @ 50 Hz)
PCA_MIN_SAMPLES = 30       # 实时 PCA 投影所需最小样本数
PCA_WINDOW = 400           # 空闲预览时滚动窗口长度 (~8 s @ 50 Hz)
DISPLAY_HZ = 8.0           # 显示刷新频率


@dataclass
class Sample:
    """一条已解析的遥测样本 (固件 7 字段)."""

    angleX: float
    angleY: float
    angleZ: float
    gyroX: float
    gyroY: float
    gyroZ: float
    time: int


@dataclass
class CollectorState:
    """主循环共享状态."""

    recording: bool = False
    quit: bool = False
    last_sample: Sample | None = None
    total_rx: int = 0                       # 串口收到的有效样本总数
    recorded: list[Sample] = field(default_factory=list)
    dropped_nonmonotonic: int = 0           # 因 time 非递增丢弃的样本数
    parse_errors: int = 0                   # 解析失败的行数


# --------------------------------------------------------------------------- #
# 串口行解析
# --------------------------------------------------------------------------- #
def parse_line(line: str) -> Sample | None:
    """解析固件 ``key=value`` 逗号行为 Sample.

    容错: 字段缺失/无法转数值时返回 None (由调用方计入 parse_errors).
    """
    line = line.strip()
    if not line:
        return None
    fields: dict[str, str] = {}
    for token in line.split(","):
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        fields[key.strip()] = val.strip()

    try:
        return Sample(
            angleX=float(fields["angleX"]),
            angleY=float(fields["angleY"]),
            angleZ=float(fields["angleZ"]),
            gyroX=float(fields["gyroX"]),
            gyroY=float(fields["gyroY"]),
            gyroZ=float(fields["gyroZ"]),
            time=int(float(fields["time"])),
        )
    except (KeyError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 实时 θ 估计 (PCA 投影; 仅操作辅助)
# --------------------------------------------------------------------------- #
def estimate_theta_pca(angle_xy: np.ndarray) -> float | None:
    """对 [angleX, angleY] 做 PCA 第一主轴投影, 返回当前 θ_phys (deg).

    对 2 列数据, 第一主轴 = 2x2 协方差矩阵最大特征值对应的特征向量,
    存在闭式解 ``phi = 0.5 * atan2(2*Cxy, Cxx - Cyy)``. 这里用纯逐元素
    运算实现 (不调用 LAPACK/BLAS), 在实时刷新路径下更轻量, 结果与
    ``raw_preprocess._principal_axis_from_angle_xy`` 的 SVD 第一主轴等价.

    符号约定与管线一致: 绝对偏离最大的样本投影到正方向, 保证激励峰
    落在 θ>90.

    注意: 这是基于当前缓冲的近似, 主轴/符号会随数据增长漂移.
    权威 θ 由管线对整段数据复算.
    """
    if angle_xy.shape[0] < PCA_MIN_SAMPLES:
        return None

    x = angle_xy[:, 0]
    y = angle_xy[:, 1]
    x = x - x.mean()
    y = y - y.mean()

    cxx = float(np.mean(x * x))
    cyy = float(np.mean(y * y))
    cxy = float(np.mean(x * y))

    # 退化: 几乎无方差
    if cxx + cyy <= 1e-12:
        return None

    phi = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
    ax = float(np.cos(phi))
    ay = float(np.sin(phi))

    # 投影 (逐元素, 不用 matmul)
    proj = x * ax + y * ay
    idx_max_dev = int(np.argmax(np.abs(proj)))
    if proj[idx_max_dev] < 0.0:
        proj = -proj

    # 当前 (最新样本) 相对均值的投影
    return 90.0 + float(proj[-1])


# --------------------------------------------------------------------------- #
# 文件命名与落盘
# --------------------------------------------------------------------------- #
def next_run_id(runs_dir: Path, day: str) -> str:
    """扫描 runs_dir, 返回当天 RYYYYMMDD_## 的下一个序号 id."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"R{day}_"
    max_idx = 0
    for p in runs_dir.glob(f"{prefix}*.txt"):
        suffix = p.stem[len(prefix):]
        if suffix.isdigit():
            max_idx = max(max_idx, int(suffix))
    return f"{prefix}{max_idx + 1:02d}"


def enforce_monotonic(samples: list[Sample]) -> tuple[list[Sample], int]:
    """保留 time 严格递增的样本, 返回 (保留列表, 丢弃数).

    在落盘前执行, 确保写出的 time 一定严格递增 (管线硬性要求).
    """
    kept: list[Sample] = []
    dropped = 0
    last_t: int | None = None
    for s in samples:
        if last_t is None or s.time > last_t:
            kept.append(s)
            last_t = s.time
        else:
            dropped += 1
    return kept, dropped


def write_pipeline_txt(path: Path, samples: list[Sample]) -> None:
    """写管线 5 列文件: 带表头, 空白(单空格)分隔."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(" ".join(PIPELINE_COLS) + "\n")
        for s in samples:
            f.write(
                f"{s.angleX:.2f} {s.angleY:.2f} "
                f"{s.gyroX:.2f} {s.gyroY:.2f} {s.time}\n"
            )


def write_full_backup(path: Path, samples: list[Sample]) -> None:
    """写完整原始备份 CSV (含 Z 轴), 便于事后排查."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FULL_COLS)
        for s in samples:
            writer.writerow(
                [
                    f"{s.angleX:.2f}",
                    f"{s.angleY:.2f}",
                    f"{s.angleZ:.2f}",
                    f"{s.gyroX:.2f}",
                    f"{s.gyroY:.2f}",
                    f"{s.gyroZ:.2f}",
                    s.time,
                ]
            )


# --------------------------------------------------------------------------- #
# 串口选择
# --------------------------------------------------------------------------- #
def choose_port(explicit: str | None) -> str:
    """返回要使用的串口名. explicit 优先; 否则列出可用口供选择."""
    if explicit:
        return explicit

    ports = list(list_ports.comports())
    if not ports:
        sys.stderr.write("未发现任何串口. 请检查设备连接, 或用 --port 指定.\n")
        sys.exit(1)

    if len(ports) == 1:
        only = ports[0]
        print(f"检测到唯一串口: {only.device} ({only.description})")
        return only.device

    print("可用串口:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device:<8} {p.description}")
    while True:
        raw = input(f"选择串口编号 [0-{len(ports) - 1}]: ").strip()
        if raw.isdigit() and 0 <= int(raw) < len(ports):
            return ports[int(raw)].device
        print("输入无效, 请重试.")


# --------------------------------------------------------------------------- #
# 读线程
# --------------------------------------------------------------------------- #
def reader_thread(
    ser: serial.Serial,
    out_q: "queue.Queue[Sample]",
    stop_evt: threading.Event,
    err_counter: list[int],
) -> None:
    """持续读取串口, 按行解析并入队. 解析失败计入 err_counter[0]."""
    buf = bytearray()
    while not stop_evt.is_set():
        try:
            chunk = ser.read(256)
        except serial.SerialException:
            break
        if chunk:
            buf.extend(chunk)
            while b"\n" in buf:
                raw_line, _, rest = buf.partition(b"\n")
                buf = bytearray(rest)
                try:
                    text = raw_line.decode("ascii", errors="ignore")
                except Exception:
                    err_counter[0] += 1
                    continue
                sample = parse_line(text)
                if sample is None:
                    if text.strip():
                        err_counter[0] += 1
                    continue
                out_q.put(sample)
        else:
            # 无数据时让出 CPU
            _time.sleep(0.005)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def format_status(state: CollectorState, theta_pca: float | None) -> str:
    s = state.last_sample
    rec = "● REC " if state.recording else "○ idle"
    if s is None:
        return f"[{rec}] 等待数据..."

    if theta_pca is not None:
        theta_str = f"θ≈{theta_pca:6.2f}°"
    else:
        theta_str = "θ≈  --  "

    n = len(state.recorded) if state.recording else state.total_rx
    n_label = "rec" if state.recording else "rx"
    return (
        f"[{rec}] {theta_str} | "
        f"angleX={s.angleX:7.2f} angleY={s.angleY:7.2f} | "
        f"gyroX={s.gyroX:7.2f} gyroY={s.gyroY:7.2f} | "
        f"{n_label}={n:<6d} drop={state.dropped_nonmonotonic}"
    )


def handle_keys(state: CollectorState) -> str | None:
    """非阻塞读取按键, 返回触发的动作: 'toggle' / 'quit' / None."""
    if not _HAS_MSVCRT:
        return None
    action = None
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch == " ":
            action = "toggle"
        elif ch in ("q", "Q"):
            return "quit"
    return action


def finalize_recording(state: CollectorState, runs_dir: Path, backup_dir: Path) -> None:
    """录制结束: 校验 -> 命名 -> 落盘 (主文件 + 完整备份)."""
    raw_samples = state.recorded
    kept, dropped = enforce_monotonic(raw_samples)
    state.dropped_nonmonotonic += dropped

    print()  # 结束单行刷新
    print("-" * 60)
    print(f"录制结束: 收到 {len(raw_samples)} 样本, "
          f"time 非递增丢弃 {dropped}, 有效 {len(kept)}.")

    if len(kept) < MIN_SAMPLES:
        print(f"⚠ 有效样本 {len(kept)} < 阈值 {MIN_SAMPLES} (~10 s). "
              f"疑似误触.")
        ans = input("仍要保存吗? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            print("已丢弃本次录制.")
            return

    day = datetime.now().strftime("%Y%m%d")
    run_id = next_run_id(runs_dir, day)
    txt_path = runs_dir / f"{run_id}.txt"
    backup_path = backup_dir / f"{run_id}_full.csv"

    write_pipeline_txt(txt_path, kept)
    write_full_backup(backup_path, kept)

    dur = (kept[-1].time - kept[0].time) / 1000.0 if len(kept) >= 2 else 0.0
    theta0 = estimate_theta_pca(
        np.array([[s.angleX, s.angleY] for s in kept], dtype=float)
    )
    print(f"✓ 已保存管线文件: {txt_path}")
    print(f"✓ 已保存完整备份: {backup_path}")
    print(f"  样本 {len(kept)}, 时长 {dur:.1f} s, "
          f"末态 θ≈{theta0:.2f}°" if theta0 is not None else
          f"  样本 {len(kept)}, 时长 {dur:.1f} s")
    print("-" * 60)


def run(port: str, baud: int) -> None:
    runs_dir = RUNS_DIR
    backup_dir = BACKUP_DIR

    print(f"打开串口 {port} @ {baud} 8N1 ...")
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
    except serial.SerialException as exc:
        sys.stderr.write(f"打开串口失败: {exc}\n")
        sys.exit(1)

    out_q: "queue.Queue[Sample]" = queue.Queue()
    stop_evt = threading.Event()
    err_counter = [0]
    thread = threading.Thread(
        target=reader_thread, args=(ser, out_q, stop_evt, err_counter), daemon=True
    )
    thread.start()

    state = CollectorState()
    # 实时 PCA 用的滚动缓冲 (空闲预览); 录制时改用全量 recorded
    preview_buf: list[tuple[float, float]] = []

    print("=" * 60)
    print("Free-decay 采集就绪.")
    print("  空格 = 开始/停止录制   q = 退出")
    print("  实时 θ 为 PCA 近似(操作辅助), 权威值由管线复算.")
    print("=" * 60)
    if not _HAS_MSVCRT:
        print("⚠ 非 Windows 环境, 按键控制不可用; 仅显示数据.")

    last_display = 0.0
    display_interval = 1.0 / DISPLAY_HZ

    try:
        while not state.quit:
            # 1) 排空队列
            drained = 0
            while True:
                try:
                    sample = out_q.get_nowait()
                except queue.Empty:
                    break
                state.last_sample = sample
                state.total_rx += 1
                drained += 1
                if state.recording:
                    state.recorded.append(sample)
                else:
                    preview_buf.append((sample.angleX, sample.angleY))
                    if len(preview_buf) > PCA_WINDOW:
                        del preview_buf[: len(preview_buf) - PCA_WINDOW]

            # 2) 按键
            action = handle_keys(state)
            if action == "quit":
                if state.recording:
                    # 退出前先收尾当前录制
                    finalize_recording(state, runs_dir, backup_dir)
                    state.recording = False
                state.quit = True
                break
            if action == "toggle":
                if not state.recording:
                    state.recorded = []
                    state.dropped_nonmonotonic = 0
                    state.recording = True
                    print("\n>>> 开始录制 <<<")
                else:
                    state.recording = False
                    finalize_recording(state, runs_dir, backup_dir)

            # 3) 显示节流
            now = _time.monotonic()
            if now - last_display >= display_interval:
                if state.recording:
                    arr = np.array(
                        [[s.angleX, s.angleY] for s in state.recorded], dtype=float
                    ) if state.recorded else np.empty((0, 2))
                else:
                    arr = np.array(preview_buf, dtype=float) if preview_buf \
                        else np.empty((0, 2))
                theta_pca = estimate_theta_pca(arr) if arr.shape[0] else None
                sys.stdout.write("\r" + format_status(state, theta_pca) + "   ")
                sys.stdout.flush()
                last_display = now

            if drained == 0:
                _time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C.")
        if state.recording:
            finalize_recording(state, runs_dir, backup_dir)
    finally:
        stop_evt.set()
        thread.join(timeout=1.0)
        ser.close()
        print(f"\n串口已关闭. 解析失败行数: {err_counter[0]}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free-decay 实验串口采集 (USART1 @ 19200, 8N1)."
    )
    parser.add_argument(
        "--port", default=None,
        help="串口名 (如 COM5). 不指定则启动时列出可用串口供选择.",
    )
    parser.add_argument(
        "--baud", type=int, default=DEFAULT_BAUD,
        help=f"波特率 (默认 {DEFAULT_BAUD}).",
    )
    args = parser.parse_args()

    port = choose_port(args.port)
    run(port, args.baud)


if __name__ == "__main__":
    main()
