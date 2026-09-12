# -*- coding: utf-8 -*-
"""
鸣潮抽卡模拟器 —— 命令行/交互式入口（含概率分布图输出）。

用法：
    python gacha_cli.py                       # 交互式
    python gacha_cli.py --feature 1 --banner character --pulls 200 --ups 2
    python gacha_cli.py --feature 3 --mode prob --c-ups 1 --w-ups 1 --pulls 200

网页前端请运行：python app.py（或打包好的 鸣潮抽卡模拟器.exe）
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gacha_sim import (
    DEFAULT_TRIALS,
    MAX_PITY,
    simulate_feature1,
    simulate_feature2,
    simulate_combined,
    exact_feature1,
    exact_feature2,
    exact_combined_expectation,
    exact_combined_probability,
)

# 中文字体（Windows 下常见中文字体，避免图上中文变方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def plot_feature1(banner, n_pulls, prob, kth, path, k_ups=1):
    trials = len(kth)
    counts = np.bincount(kth, minlength=n_pulls + 2)  # 索引0..N+1
    pmf = counts[1:] / trials  # pmf[k-1]=第k抽凑齐；pmf[N]=N抽内未凑齐
    cdf = np.cumsum(pmf[:-1])  # cdf[k-1]=k抽内凑齐概率，k=1..N

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    banner_cn = "角色池" if banner == "character" else "武器池"
    up_label = "首次出当期UP" if k_ups == 1 else f"凑齐第 {k_ups} 个当期UP"

    x = np.arange(1, n_pulls + 1)
    ax = axes[0]
    ax.bar(x, pmf[:-1], color="#4C72B0", width=0.9, label=up_label)
    ax.bar([n_pulls + 1], [pmf[-1]], color="#C44E52", width=0.9, label=f"{n_pulls}抽内未凑齐")
    ax.set_xlabel("抽数 k")
    ax.set_ylabel("概率")
    ax.set_title(f"{banner_cn}：{up_label} 的概率分布（{trials:,} 次模拟）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(x, cdf, color="#55A868", marker="o", markersize=2.5, linewidth=1.5)
    ax.axvline(n_pulls, color="#C44E52", linestyle="--", linewidth=1.2)
    ax.text(n_pulls, cdf[-1] if n_pulls else 0,
            f"  P(≤{n_pulls}抽) = {prob * 100:.2f}%", color="#C44E52", fontweight="bold")
    ax.set_xlabel("抽数 k")
    ax.set_ylabel("累计概率")
    ax.set_title(f"累计概率（CDF）：{n_pulls} 抽内凑齐 {k_ups} 个当期UP 的概率")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature2(banner, k_ups, pulls, mean, path):
    trials = len(pulls)
    banner_cn = "角色池" if banner == "character" else "武器池"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(pulls, bins=80, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax.axvline(mean, color="#C44E52", linestyle="--", linewidth=2,
               label=f"期望 {mean:.2f} 抽")
    med = float(np.median(pulls))
    std = float(np.std(pulls))
    ax.text(0.97, 0.95,
            f"模拟 {trials:,} 次\n期望(平均) = {mean:.2f} 抽\n中位数 = {med:.1f} 抽\n标准差 = {std:.2f} 抽",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("需要的抽数")
    ax.set_ylabel("频数")
    ax.set_title(f"{banner_cn}：抽出 {k_ups} 个当期UP 所需抽数分布")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature3(mode, n_pulls, c_ups, w_ups, prob, total, mean, path):
    trials = len(total)
    desc = f"{c_ups} 角色 + {w_ups} 武器"
    if mode == "prob":
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        sorted_total = np.sort(total)
        ks = np.arange(1, n_pulls + 1)
        cdf = np.searchsorted(sorted_total, ks, side="right") / trials
        ax = axes[0]
        ax.plot(ks, cdf, color="#55A868", linewidth=1.5)
        ax.axvline(n_pulls, color="#C44E52", linestyle="--", linewidth=1.2)
        ax.text(n_pulls, cdf[-1], f"  P(≤{n_pulls}总抽) = {prob * 100:.2f}%",
                color="#C44E52", fontweight="bold")
        ax.set_xlabel("总抽数 k")
        ax.set_ylabel("累计概率")
        ax.set_title(f"累计概率 CDF：{n_pulls} 总抽内抽齐（{desc}）")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax = axes[1]
        ax.hist(total, bins=80, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(n_pulls, color="#C44E52", linestyle="--", linewidth=2, label=f"预算 {n_pulls} 抽")
        ax.set_xlabel("总抽数")
        ax.set_ylabel("频数")
        ax.set_title(f"总抽数分布（抽齐 {desc}，模拟 {trials:,} 次）")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.hist(total, bins=80, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(mean, color="#C44E52", linestyle="--", linewidth=2, label=f"期望 {mean:.2f} 抽")
        med = float(np.median(total))
        std = float(np.std(total))
        ax.text(0.97, 0.95,
                f"模拟 {trials:,} 次\n期望(平均) = {mean:.2f} 抽\n中位数 = {med:.1f} 抽\n标准差 = {std:.2f} 抽",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        ax.set_xlabel("总抽数")
        ax.set_ylabel("频数")
        ax.set_title(f"抽齐（{desc}）所需总抽数分布")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_feature1(args):
    banner, n_pulls = args.banner, args.pulls
    k_ups = getattr(args, "ups", 1) or 1
    guaranteed, pity = bool(args.guaranteed), args.pity
    trials = args.trials
    print("=" * 62)
    print(f"[功能1] {('角色池' if banner == 'character' else '武器池')} "
          f"{n_pulls} 抽内抽齐 {k_ups} 个当期UP 的概率")
    print(f"        初始：{'是' if guaranteed else '否'}大保底，已垫 {pity} 抽，模拟 {trials:,} 次")
    print("=" * 62)

    prob, kth = simulate_feature1(banner, n_pulls, guaranteed, pity, trials, args.seed, k_ups=k_ups)
    exact = exact_feature1(banner, n_pulls, guaranteed, pity, k_ups=k_ups)
    se = np.sqrt(prob * (1 - prob) / trials)

    print(f"  蒙特卡洛结果：P(在 {n_pulls} 抽内抽齐 {k_ups} 个当期UP) = {prob * 100:.4f}%")
    print(f"  解析精确值  ：P(在 {n_pulls} 抽内抽齐 {k_ups} 个当期UP) = {exact * 100:.4f}%")
    print(f"  （蒙特卡洛标准误 ±{se * 100:.4f}%）")

    out = os.path.join(args.out, f"feature1_{banner}_n{n_pulls}_k{k_ups}_g{int(guaranteed)}_p{pity}.png")
    plot_feature1(banner, n_pulls, prob, kth, out, k_ups=k_ups)
    print(f"  概率分布图已保存：{out}\n")


def run_feature2(args):
    banner, k_ups = args.banner, args.ups
    guaranteed, pity = bool(args.guaranteed), args.pity
    trials = args.trials
    print("=" * 62)
    print(f"[功能2] {('角色池' if banner == 'character' else '武器池')} "
          f"抽出 {k_ups} 个当期UP 所需抽数期望")
    print(f"        初始：{'是' if guaranteed else '否'}大保底，已垫 {pity} 抽，模拟 {trials:,} 次")
    print("=" * 62)

    pulls = simulate_feature2(banner, k_ups, guaranteed, pity, trials, args.seed)
    mean = float(pulls.mean())
    exact = exact_feature2(banner, k_ups, guaranteed, pity)
    se = float(pulls.std()) / np.sqrt(trials)

    print(f"  蒙特卡洛结果：期望 = {mean:.4f} 抽（还需的抽数）")
    print(f"  解析精确值  ：期望 = {exact:.4f} 抽")
    print(f"  （蒙特卡洛标准误 ±{se:.4f} 抽）")

    out = os.path.join(args.out, f"feature2_{banner}_k{k_ups}_g{int(guaranteed)}_p{pity}.png")
    plot_feature2(banner, k_ups, pulls, mean, out)
    print(f"  概率分布图已保存：{out}\n")


def run_feature3(args):
    mode = args.mode  # 'prob' | 'exp'
    c_ups, w_ups = args.c_ups, args.w_ups
    guar_c = bool(args.guaranteed)
    pity_c, pity_w = args.pity_c, args.pity_w
    trials = args.trials
    print("=" * 62)
    print(f"[功能3 合计版] {c_ups} 角色 + {w_ups} 武器"
          f"（{'抽出概率' if mode == 'prob' else '抽数期望'}）")
    print(f"        角色池：{'是' if guar_c else '否'}大保底，垫 {pity_c} 抽；"
          f"武器池：垫 {pity_w} 抽；模拟 {trials:,} 次")
    print("=" * 62)

    total = simulate_combined(c_ups, w_ups, guar_c, pity_c, pity_w, trials, args.seed)
    mean = float(total.mean())
    exp_exact = exact_combined_expectation(c_ups, w_ups, guar_c, pity_c, pity_w)

    if mode == "prob":
        n_pulls = args.pulls
        prob = float((total <= n_pulls).mean())
        exact = exact_combined_probability(n_pulls, c_ups, w_ups, guar_c, pity_c, pity_w)
        se = float(np.sqrt(prob * (1 - prob) / trials))
        print(f"  蒙特卡洛结果：P(在 {n_pulls} 总抽内抽齐) = {prob * 100:.4f}%")
        print(f"  解析精确值  ：P(在 {n_pulls} 总抽内抽齐) = {exact * 100:.4f}%")
        print(f"  期望总抽数：{mean:.4f}（精确 {exp_exact:.4f}），标准误 ±{se * 100:.4f}%")
        out = os.path.join(args.out,
                           f"feature3_prob_c{c_ups}w{w_ups}_n{n_pulls}_gc{int(guar_c)}_pc{pity_c}_pw{pity_w}.png")
    else:
        se = float(total.std()) / np.sqrt(trials)
        print(f"  蒙特卡洛结果：期望总抽数 = {mean:.4f} 抽")
        print(f"  解析精确值  ：期望总抽数 = {exp_exact:.4f} 抽")
        print(f"  （蒙特卡洛标准误 ±{se:.4f} 抽）")
        out = os.path.join(args.out,
                           f"feature3_exp_c{c_ups}w{w_ups}_gc{int(guar_c)}_pc{pity_c}_pw{pity_w}.png")
        n_pulls = None
        prob = None

    plot_feature3(mode, n_pulls, c_ups, w_ups, prob, total, mean, out)
    print(f"  概率分布图已保存：{out}\n")


def interactive():
    """无命令行参数时的交互式模式。"""
    print("=" * 62)
    print("鸣潮抽卡模拟器")
    print("=" * 62)
    print("选择功能：")
    print("  1) 计算 N 抽内抽齐 K 个当期UP 的概率")
    print("  2) 计算抽出 K 个当期UP 所需抽数期望")
    print("  3) 武器+角色 合计版（抽出概率 / 抽数期望）")
    choice = input("请输入功能编号 [1/2/3]：").strip()
    if choice not in ("1", "2", "3"):
        print("无效选择，退出。")
        return

    banner, guaranteed, pity = "character", False, 0
    if choice in ("1", "2"):
        banner = input("抽卡池 [character/weapon，默认 character]：").strip().lower()
        if banner == "":
            banner = "character"
        if banner not in ("character", "weapon"):
            print("无效池，使用 character。")
            banner = "character"
        guaranteed = input("下一个五星是否为大保底(必为当期UP)？[y/N]：").strip().lower() in ("y", "yes", "1")
        pity_s = input("已垫多少抽(0~78，默认 0)：").strip()
        pity = int(pity_s) if pity_s else 0
        pity = min(max(pity, 0), MAX_PITY)

    args = argparse.Namespace(
        banner=banner, guaranteed=guaranteed, pity=pity,
        trials=DEFAULT_TRIALS, out=os.getcwd(), seed=None,
    )
    if choice == "1":
        n_pulls = int(input("请输入抽数 N：").strip())
        k_ups = int(input("请输入需要的当期UP 数量 K（默认 1）：").strip() or "1")
        args.pulls, args.ups = n_pulls, k_ups
        run_feature1(args)
    elif choice == "2":
        k_ups = int(input("请输入需要的当期UP 数量 K：").strip())
        args.ups = k_ups
        run_feature2(args)
    else:
        c_ups = int(input("角色 UP 数量 C（默认 1）：").strip() or "1")
        w_ups = int(input("武器 UP 数量 W（默认 1）：").strip() or "1")
        mode = input("合计模式 [prob/exp，默认 prob]：").strip().lower() or "prob"
        guar = input("角色池：下一个五星是否为大保底？[y/N]：").strip().lower() in ("y", "yes", "1")
        pc = int(input("角色池已垫抽数（默认 0）：").strip() or "0")
        pw = int(input("武器池已垫抽数（默认 0）：").strip() or "0")
        args.c_ups, args.w_ups, args.mode = c_ups, w_ups, mode
        args.guaranteed = guar
        args.pity_c, args.pity_w = min(max(pc, 0), MAX_PITY), min(max(pw, 0), MAX_PITY)
        args.pulls = int(input("总抽数预算 N：").strip()) if mode == "prob" else None
        run_feature3(args)


def parse_args(argv):
    p = argparse.ArgumentParser(description="鸣潮抽卡模拟器（蒙特卡洛 + 解析校验）")
    p.add_argument("--feature", choices=["1", "2", "3"],
                   help="功能：1=概率，2=抽数期望，3=武器+角色合计（省略则进入交互模式）")
    p.add_argument("--banner", choices=["character", "weapon"], default="character",
                   help="抽卡池：character=角色池(50%%小保底)，weapon=武器池(100%%当期)")
    p.add_argument("--pulls", type=int, help="功能1/功能3(prob)：抽数 N")
    p.add_argument("--ups", type=int, help="功能1/2：需要的当期UP 数量 K")
    p.add_argument("--mode", choices=["prob", "exp"], default="prob",
                   help="功能3：prob=抽出概率，exp=抽数期望")
    p.add_argument("--c-ups", type=int, default=1, help="功能3：角色 UP 数量 C")
    p.add_argument("--w-ups", type=int, default=1, help="功能3：武器 UP 数量 W")
    p.add_argument("--guaranteed", action="store_true", help="下一个五星是否为大保底(必为当期UP)")
    p.add_argument("--pity", type=int, default=0, help="已垫抽数(0~78，默认 0)")
    p.add_argument("--pity-c", type=int, default=0, help="功能3：角色池已垫抽数")
    p.add_argument("--pity-w", type=int, default=0, help="功能3：武器池已垫抽数")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help=f"模拟次数(默认 {DEFAULT_TRIALS})")
    p.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)), help="图片输出目录")
    p.add_argument("--seed", type=int, default=None, help="随机种子(可复现，默认随机)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.pity = min(max(args.pity, 0), MAX_PITY)
    os.makedirs(args.out, exist_ok=True)

    if args.feature is None:
        interactive()
        return

    if args.feature == "1":
        if args.pulls is None or args.pulls <= 0:
            print("功能1 需要 --pulls N（N>0）。")
            sys.exit(2)
        run_feature1(args)
    elif args.feature == "2":
        if args.ups is None or args.ups <= 0:
            print("功能2 需要 --ups K（K>0）。")
            sys.exit(2)
        run_feature2(args)
    else:  # feature 3 合计版
        args.c_ups = max(0, args.c_ups)
        args.w_ups = max(0, args.w_ups)
        args.pity_c = min(max(args.pity_c, 0), MAX_PITY)
        args.pity_w = min(max(args.pity_w, 0), MAX_PITY)
        if args.c_ups == 0 and args.w_ups == 0:
            print("功能3 需要至少一个 UP（--c-ups 或 --w-ups > 0）。")
            sys.exit(2)
        if args.mode == "prob" and (args.pulls is None or args.pulls <= 0):
            print("功能3(prob) 需要 --pulls N（N>0）。")
            sys.exit(2)
        run_feature3(args)


if __name__ == "__main__":
    main()
