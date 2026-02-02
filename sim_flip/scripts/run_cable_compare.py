from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt

from sim_flip.src.dynamics import load_params_yaml, simulate


def _run_case(params: dict, label: str):
    result = simulate(
        y0=[0.0, 0.0, 0.0, 0.0],
        t_span=(0.0, 20.0),
        dt_out=0.01,
        params=params,
    )
    df = result.data
    t_end = float(df["t"].iloc[-1]) if len(df) else float("nan")
    return df, label, result, t_end


def _metrics(df):
    t = df["t"].to_numpy()
    theta = df["theta_deg"].to_numpy()

    above_80 = t[theta >= 80.0]
    t_80 = float(above_80[0]) if len(above_80) else float("nan")

    theta_final = float(theta[-1])
    return theta_final, t_80


def _final_snapshot(df):
    row = df.iloc[-1]
    return {
        "theta_deg": float(row["theta_deg"]),
        "V": float(row["V"]),
        "Q": float(row["Q"]),
        "alpha_raw_deg": float(row["alpha_raw_deg"]),
        "alpha_deg": float(row["alpha_deg"]),
        "M_cfd": float(row["M_cfd"]),
        "M_bg": float(row["M_bg"]),
        "M_damp": float(row["M_damp"]),
    }


def main() -> None:
    params_path = Path(__file__).resolve().parents[1] / "configs" / "params_nominal.yaml"
    base_params = load_params_yaml(params_path)

    params_off = copy.deepcopy(base_params)
    params_off["cable"]["enabled"] = False
    params_off["cable"]["K_cable"] = 0.0

    params_on = copy.deepcopy(base_params)
    params_on["cable"]["enabled"] = True

    df_off, label_off, res_off, t_end_off = _run_case(params_off, "Cable OFF")
    df_on, label_on, res_on, t_end_on = _run_case(params_on, "Cable ON")

    theta_final_off, t_80_off = _metrics(df_off)
    theta_final_on, t_80_on = _metrics(df_on)

    snap_off = _final_snapshot(df_off)
    snap_on = _final_snapshot(df_on)

    print(
        f"{label_off}: theta_final={theta_final_off:.3f} deg, t_80={t_80_off:.3f} s, "
        f"t_end={t_end_off:.3f} s, success={res_off.success}, status={res_off.status}"
    )
    print(
        f"  final snapshot: theta={snap_off['theta_deg']:.3f} deg, V={snap_off['V']:.3f} m/s, "
        f"Q={snap_off['Q']:.3f}, alpha_raw={snap_off['alpha_raw_deg']:.3f} deg, "
        f"alpha_lut={snap_off['alpha_deg']:.3f} deg, M_cfd={snap_off['M_cfd']:.5f}, "
        f"M_bg={snap_off['M_bg']:.5f}, M_damp={snap_off['M_damp']:.5f}"
    )
    print(
        f"{label_on}:  theta_final={theta_final_on:.3f} deg, t_80={t_80_on:.3f} s, "
        f"t_end={t_end_on:.3f} s, success={res_on.success}, status={res_on.status}"
    )
    print(
        f"  final snapshot: theta={snap_on['theta_deg']:.3f} deg, V={snap_on['V']:.3f} m/s, "
        f"Q={snap_on['Q']:.3f}, alpha_raw={snap_on['alpha_raw_deg']:.3f} deg, "
        f"alpha_lut={snap_on['alpha_deg']:.3f} deg, M_cfd={snap_on['M_cfd']:.5f}, "
        f"M_bg={snap_on['M_bg']:.5f}, M_damp={snap_on['M_damp']:.5f}"
    )

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(df_off["t"], df_off["theta_deg"], label=label_off)
    plt.plot(df_on["t"], df_on["theta_deg"], label=label_on)
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (deg)")
    plt.title("Cable OFF vs ON: Theta(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / "theta_cable_compare.png"
    plt.savefig(plot_path, dpi=160)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
