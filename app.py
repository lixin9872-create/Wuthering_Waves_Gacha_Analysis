# -*- coding: utf-8 -*-
"""
鸣潮抽卡模拟器 —— 可交互 Web 前端（后端）。

只使用 Python 标准库（http.server），复用 gacha_sim.py 里已验证的模拟逻辑，
无需额外 pip 安装。运行后自动打开浏览器：

    python app.py

访问地址：http://127.0.0.1:8000
"""

import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

import gacha_sim  # 复用已验证的模拟逻辑（出金概率、蒙特卡洛、动态规划校验）

if getattr(sys, "frozen", False):  # PyInstaller 打包运行：资源在解包临时目录
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PORT = 8000


class Handler(BaseHTTPRequestHandler):
    server_version = "GachaSim/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.address_string(), fmt % args))

    # ------------------------------------------------------------------ #
    # 基础响应
    # ------------------------------------------------------------------ #
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, filepath, ctype):
        if not os.path.isfile(filepath):
            return self._send(404, json.dumps({"error": "文件不存在"}))
        with open(filepath, "rb") as f:
            return self._send(200, f.read(), ctype)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(BASE_DIR, "index.html"), "text/html; charset=utf-8")
        if path == "/api/simulate":
            return self.handle_simulate(parse_qs(parsed.query))
        if path.startswith("/static/"):
            name = os.path.basename(path)  # 用 basename 防止目录穿越
            return self._serve_file(os.path.join(BASE_DIR, "static", name), "application/javascript; charset=utf-8")

        return self._send(404, json.dumps({"error": "未找到路径"}))

    # ------------------------------------------------------------------ #
    # 模拟接口
    # ------------------------------------------------------------------ #
    def handle_simulate(self, q):
        try:
            feature = q.get("feature", ["1"])[0]
            banner = q.get("banner", ["character"])[0]
            pity = int(q.get("pity", ["0"])[0])
            guaranteed = q.get("guaranteed", ["0"])[0].lower() in ("1", "true", "yes", "on")
            trials = int(q.get("trials", [str(gacha_sim.DEFAULT_TRIALS)])[0])

            trials = max(1, min(trials, 1_000_000))
            pity = max(0, min(pity, gacha_sim.MAX_PITY))
            if banner not in ("character", "weapon"):
                raise ValueError("无效的抽卡池类型")

            if feature == "1":
                n_pulls = int(q.get("pulls", ["80"])[0])
                k_ups = int(q.get("ups", ["1"])[0])
                if n_pulls <= 0:
                    raise ValueError("抽数 N 必须 > 0")
                if k_ups <= 0:
                    raise ValueError("UP 数量 K 必须 > 0")
                prob, kth = gacha_sim.simulate_feature1(banner, n_pulls, guaranteed, pity, trials, k_ups=k_ups)
                counts = np.bincount(kth, minlength=n_pulls + 2)
                pmf = (counts[1:] / trials).tolist()          # 长度 N+1，末位为“N抽内未抽齐”
                cdf = np.cumsum(counts[1:n_pulls + 1] / trials).tolist()  # 长度 N
                exact = gacha_sim.exact_feature1(banner, n_pulls, guaranteed, pity, k_ups=k_ups)
                se = float(np.sqrt(prob * (1 - prob) / trials))
                return self._send(200, json.dumps({
                    "feature": 1, "banner": banner, "n_pulls": n_pulls, "k_ups": k_ups,
                    "pity": pity, "guaranteed": guaranteed, "trials": trials,
                    "prob": float(prob), "exact": exact, "se": se,
                    "pmf": pmf, "cdf": cdf,
                }))

            if feature == "2":
                k_ups = int(q.get("ups", ["1"])[0])
                if k_ups <= 0:
                    raise ValueError("UP 数量 K 必须 > 0")
                pulls = gacha_sim.simulate_feature2(banner, k_ups, guaranteed, pity, trials)
                mean = float(pulls.mean())
                median = float(np.median(pulls))
                std = float(pulls.std())
                exact = gacha_sim.exact_feature2(banner, k_ups, guaranteed, pity)
                se = float(pulls.std() / np.sqrt(trials))
                hist, edges = np.histogram(pulls, bins=60)
                # 累计概率（模拟 + 解析精确值）
                kmax = int(pulls.max())
                sorted_pulls = np.sort(pulls)
                ks = np.arange(1, kmax + 1)
                cdf = (np.searchsorted(sorted_pulls, ks, side="right") / trials).tolist()
                exact_cdf = gacha_sim.exact_cdf_k(banner, kmax, k_ups, guaranteed, pity)[1:].tolist()
                return self._send(200, json.dumps({
                    "feature": 2, "banner": banner, "k_ups": k_ups,
                    "pity": pity, "guaranteed": guaranteed, "trials": trials,
                    "mean": mean, "exact": exact, "se": se,
                    "median": median, "std": std,
                    "hist_counts": hist.tolist(), "hist_edges": edges.tolist(),
                    "cdf": cdf, "exact_cdf": exact_cdf,
                }))

            if feature == "3":
                mode = q.get("mode", ["prob"])[0]
                c_ups = int(q.get("c_ups", ["1"])[0])
                w_ups = int(q.get("w_ups", ["1"])[0])
                pity_c = int(q.get("pity_c", ["0"])[0])
                pity_w = int(q.get("pity_w", ["0"])[0])
                c_ups = max(0, min(c_ups, 100))
                w_ups = max(0, min(w_ups, 100))
                pity_c = max(0, min(pity_c, gacha_sim.MAX_PITY))
                pity_w = max(0, min(pity_w, gacha_sim.MAX_PITY))
                if c_ups == 0 and w_ups == 0:
                    raise ValueError("角色与武器数量不能都为 0")

                if mode == "prob":
                    n_pulls = int(q.get("pulls", ["200"])[0])
                    if n_pulls <= 0:
                        raise ValueError("总抽数 N 必须 > 0")
                    total = gacha_sim.simulate_combined(c_ups, w_ups, guaranteed, pity_c, pity_w, trials)
                    prob = float((total <= n_pulls).mean())
                    exact = gacha_sim.exact_combined_probability(n_pulls, c_ups, w_ups, guaranteed, pity_c, pity_w)
                    se = float(np.sqrt(prob * (1 - prob) / trials))
                    sorted_total = np.sort(total)
                    ks = np.arange(1, n_pulls + 1)
                    cdf = (np.searchsorted(sorted_total, ks, side="right") / trials).tolist()
                    hist, edges = np.histogram(total, bins=60)
                    mean_total = float(total.mean())
                    exp_exact = gacha_sim.exact_combined_expectation(c_ups, w_ups, guaranteed, pity_c, pity_w)
                    return self._send(200, json.dumps({
                        "feature": 3, "mode": "prob", "n_pulls": n_pulls,
                        "c_ups": c_ups, "w_ups": w_ups, "pity_c": pity_c, "pity_w": pity_w,
                        "guaranteed": guaranteed, "trials": trials,
                        "prob": prob, "exact": exact, "se": se,
                        "mean": mean_total, "exp_exact": exp_exact,
                        "cdf": cdf, "hist_counts": hist.tolist(), "hist_edges": edges.tolist(),
                    }))

                if mode == "exp":
                    total = gacha_sim.simulate_combined(c_ups, w_ups, guaranteed, pity_c, pity_w, trials)
                    mean = float(total.mean())
                    median = float(np.median(total))
                    std = float(total.std())
                    exact = gacha_sim.exact_combined_expectation(c_ups, w_ups, guaranteed, pity_c, pity_w)
                    se = float(total.std() / np.sqrt(trials))
                    hist, edges = np.histogram(total, bins=60)
                    # 累计概率：模拟 CDF + 解析精确 CDF（两池独立，卷积计算）
                    kmax = int(total.max())
                    sorted_total = np.sort(total)
                    ks = np.arange(1, kmax + 1)
                    cdf = (np.searchsorted(sorted_total, ks, side="right") / trials).tolist()
                    cdf_char = gacha_sim.exact_cdf_k("character", kmax, c_ups, guaranteed, pity_c) \
                        if c_ups > 0 else np.ones(kmax + 1)
                    cdf_weapon = gacha_sim.exact_cdf_k("weapon", kmax, w_ups, False, pity_w) \
                        if w_ups > 0 else np.ones(kmax + 1)
                    pmf_weapon = np.diff(np.concatenate(([0.0], cdf_weapon)))
                    exact_cdf = np.convolve(pmf_weapon, cdf_char)[1:kmax + 1].tolist()
                    return self._send(200, json.dumps({
                        "feature": 3, "mode": "exp",
                        "c_ups": c_ups, "w_ups": w_ups, "pity_c": pity_c, "pity_w": pity_w,
                        "guaranteed": guaranteed, "trials": trials,
                        "mean": mean, "exact": exact, "se": se,
                        "median": median, "std": std,
                        "hist_counts": hist.tolist(), "hist_edges": edges.tolist(),
                        "cdf": cdf, "exact_cdf": exact_cdf,
                    }))

                raise ValueError("无效的合计模式")

            raise ValueError("无效的功能编号")
        except Exception as e:  # noqa: BLE001 —— 返回可读错误给前端
            return self._send(400, json.dumps({"error": str(e)}))


def main():
    port = int(os.environ.get("PORT", str(DEFAULT_PORT)))
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print("=" * 52)
    print("鸣潮抽卡模拟器前端已启动")
    print(f"  地址：{url}")
    print("  按 Ctrl+C 停止服务")
    print("=" * 52)

    def _open():
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    if os.environ.get("NO_BROWSER", "0") != "1":
        threading.Timer(0.6, _open).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
