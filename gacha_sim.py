# -*- coding: utf-8 -*-
"""
鸣潮（Wuthering Waves）抽卡模拟核心库（仅依赖 numpy，供 Web 前后端与命令行复用）。

概率规则（取自《抽卡概率.docx》）：
  角色池 与 武器池 的“出金(五星)概率”曲线相同：
    * 第  1 ~ 65 抽：出金概率固定 0.8%
    * 第 66 ~ 70 抽：每抽递增 4%
    * 第 71 ~ 75 抽：每抽递增 8%
    * 第 76 ~ 78 抽：每抽递增 10%
    * 第 79 抽：100% 必出金（硬保底）
  角色池：出金时，是“当期 UP 角色”的概率为 50%（小保底）；
          若不是当期 UP，则下一次出金必为当期 UP（大保底）。
  武器池：出金时，是“当期 UP 武器”的概率为 100%（无 50/50）。

  说明：原文“第71~75抽+8%”与“第75~78抽+10%”在第 75 抽处重叠。这里按最常见口径，
  将 +10% 区间视为第 76~78 抽，这样第 79 抽恰好 100%。

用法：
  命令行模式：python gacha_cli.py [参数]
  网页前端  ：python app.py        （或直接运行打包好的 鸣潮抽卡模拟器.exe）
"""

import numpy as np

DEFAULT_TRIALS = 100_000
MAX_PITY = 78  # 垫抽数合法范围 0~78；pity=78 表示下一抽必出金


# --------------------------------------------------------------------------- #
# 出金概率表
# --------------------------------------------------------------------------- #
def five_star_rate(pity: int) -> float:
    """当前已垫 pity 抽(未出金)时，这一抽出金的概率。pity=78 时为硬保底(1.0)。"""
    p = pity + 1  # 当前是本轮的第几抽
    if p <= 65:
        return 0.008
    if p <= 70:
        return 0.008 + 0.04 * (p - 65)
    if p <= 75:
        return 0.008 + 0.04 * 5 + 0.08 * (p - 70)
    if p <= 78:
        return 0.008 + 0.04 * 5 + 0.08 * 5 + 0.10 * (p - 75)
    return 1.0


RATE_TABLE = np.array([five_star_rate(p) for p in range(MAX_PITY + 1)], dtype=float)  # 索引 0~78


def featured_rate(banner: str) -> float:
    """出金时抽到“当期 UP”的概率：角色池 0.5，武器池 1.0。"""
    return 1.0 if banner == "weapon" else 0.5


# --------------------------------------------------------------------------- #
# 功能1：N 抽内抽齐 K 个当期 UP 的概率（蒙特卡洛）
# --------------------------------------------------------------------------- #
def simulate_feature1(banner, n_pulls, guaranteed, pity, trials=DEFAULT_TRIALS, seed=None, k_ups=1):
    """返回 (概率, kth_pull数组)。kth_pull[i] = 第 i 个试次凑齐第 k_ups 个当期UP 的抽数(1..N)，
    未在 N 抽内凑齐则记为 N+1。"""
    rng = np.random.default_rng(seed)
    fr = featured_rate(banner)
    pity_arr = np.full(trials, pity, dtype=np.int64)
    guar_arr = np.full(trials, bool(guaranteed), dtype=bool)
    ups = np.zeros(trials, dtype=np.int64)
    kth = np.full(trials, n_pulls + 1, dtype=np.int64)

    for k in range(1, n_pulls + 1):
        rate = RATE_TABLE[pity_arr]
        hit5 = rng.random(trials) < rate
        is_up = guar_arr | (rng.random(trials) < fr)
        got = hit5 & is_up
        # 出金则垫抽清零，否则 +1
        pity_arr = np.where(hit5, 0, pity_arr + 1)
        # 出金且不是当期UP -> 下一个五星变“大保底”；出金且是当期UP -> 恢复小保底
        guar_arr = np.where(hit5, ~is_up, guar_arr)
        ups = np.where(got, ups + 1, ups)
        # 记录恰好在这一抽凑齐 k_ups 个当期UP 的试次
        reached = got & (ups == k_ups) & (kth == n_pulls + 1)
        kth = np.where(reached, k, kth)

    prob = float((kth <= n_pulls).mean())
    return prob, kth


# --------------------------------------------------------------------------- #
# 功能2：抽出 K 个当期 UP 还需要的抽数（蒙特卡洛）
# --------------------------------------------------------------------------- #
def simulate_feature2(banner, k_ups, guaranteed, pity, trials=DEFAULT_TRIALS, seed=None):
    """返回 pulls 数组：每个试次从当前状态抽到 K 个当期UP 所需的总抽数。"""
    rng = np.random.default_rng(seed)
    fr = featured_rate(banner)
    pity_arr = np.full(trials, pity, dtype=np.int64)
    guar_arr = np.full(trials, bool(guaranteed), dtype=bool)
    ups = np.zeros(trials, dtype=np.int64)
    pulls = np.zeros(trials, dtype=np.int64)
    done = ups >= k_ups

    safety = 2000 * max(1, k_ups) + 2000
    for _ in range(safety):
        if done.all():
            break
        active = ~done
        rate = RATE_TABLE[pity_arr]
        hit5 = rng.random(trials) < rate
        is_up = guar_arr | (rng.random(trials) < fr)
        got = hit5 & is_up
        pity_arr = np.where(hit5, 0, pity_arr + 1)
        guar_arr = np.where(hit5, ~is_up, guar_arr)
        pulls += active
        ups = np.where(got, ups + 1, ups)
        done = ups >= k_ups
    else:
        raise RuntimeError("模拟迭代次数达到安全上限仍未收敛，请检查参数。")
    return pulls


# --------------------------------------------------------------------------- #
# 精确解析解（动态规划，用于交叉校验蒙特卡洛结果）
# --------------------------------------------------------------------------- #
def exact_cdf_k(banner, n_pulls, k_ups, guaranteed, pity):
    """返回长度 n_pulls+1 的数组：cdf[n] = 在 n 抽内抽齐 k_ups 个当期UP 的概率。"""
    fr = featured_rate(banner)
    K = k_ups
    # cur[p][g][r]：pity=p, 是否大保底=g, 还需 r 个当期UP 时，当前剩余抽数下抽齐的概率
    cur = np.zeros((MAX_PITY + 2, 2, K + 1))
    cur[:, :, 0] = 1.0
    cdf = np.zeros(n_pulls + 1)
    cdf[0] = 1.0 if K == 0 else 0.0
    for n in range(1, n_pulls + 1):
        nxt = np.zeros_like(cur)
        nxt[:, :, 0] = 1.0
        for g in (0, 1):
            fg = 1.0 if g else fr  # 出金时抽到当期UP的概率
            # 命中项与 p 无关，向量化 r=1..K
            hit = fg * cur[0][0][:K] + (1.0 - fg) * cur[0][1][1:]
            # 未出金：p -> p+1（p=0..78；p=79 为填充，rate=1 时该项系数为 0）
            miss = (1.0 - RATE_TABLE)[:, None] * cur[1:MAX_PITY + 2][:, g, 1:]
            nxt[:MAX_PITY + 1, g, 1:] = miss + RATE_TABLE[:, None] * hit[None, :]
        cur = nxt
        cdf[n] = cur[pity][1 if guaranteed else 0][K]
    return cdf


def exact_feature1(banner, n_pulls, guaranteed, pity, k_ups=1):
    """精确计算 N 抽内抽出至少 k_ups 个当期UP 的概率。"""
    return float(exact_cdf_k(banner, n_pulls, k_ups, guaranteed, pity)[n_pulls])


def exact_feature2(banner, k_ups, guaranteed, pity):
    """精确计算抽出 K 个当期UP 还需要的抽数期望。"""
    fr = featured_rate(banner)
    # E[p][g][r]：pity=p, 是否大保底=g, 还需 r 个当期UP 的期望抽数
    E = np.zeros((MAX_PITY + 2, 2, k_ups + 1))
    for r in range(1, k_ups + 1):
        # 必须先算大保底(g=1)，因为小保底(g=0)的命中项要引用 E[0][1][r]
        for g in (1, 0):
            for p in range(MAX_PITY, -1, -1):
                rate = five_star_rate(p)
                if g:
                    hit_term = E[0][0][r - 1]
                else:
                    hit_term = fr * E[0][0][r - 1] + (1.0 - fr) * E[0][1][r]
                miss_term = E[p + 1][g][r]
                E[p][g][r] = 1.0 + rate * hit_term + (1.0 - rate) * miss_term
    return float(E[pity][1 if guaranteed else 0][k_ups])


# --------------------------------------------------------------------------- #
# 功能3（合计版）：角色池 + 武器池 一起抽
# --------------------------------------------------------------------------- #
def simulate_combined(c_ups, w_ups, guar_c, pity_c, pity_w, trials=DEFAULT_TRIALS, seed=None):
    """返回 total_pulls 数组：每个试次抽齐 c_ups 个当期UP角色 + w_ups 个当期UP武器
    所需的总抽数（两个池独立，总抽数 = 角色抽数 + 武器抽数）。"""
    ss = np.random.SeedSequence(seed)
    child_c, child_w = ss.spawn(2)  # 两个独立随机流，保证角色/武器互不相关

    if c_ups > 0:
        char_pulls = simulate_feature2("character", c_ups, guar_c, pity_c, trials, seed=child_c)
    else:
        char_pulls = np.zeros(trials, dtype=np.int64)

    if w_ups > 0:
        weapon_pulls = simulate_feature2("weapon", w_ups, False, pity_w, trials, seed=child_w)
    else:
        weapon_pulls = np.zeros(trials, dtype=np.int64)

    return char_pulls + weapon_pulls


def exact_combined_expectation(c_ups, w_ups, guar_c, pity_c, pity_w):
    """精确计算抽齐 c_ups 个角色 + w_ups 个武器 所需总抽数期望（= 两者期望之和）。"""
    e_char = exact_feature2("character", c_ups, guar_c, pity_c) if c_ups > 0 else 0.0
    e_weapon = exact_feature2("weapon", w_ups, False, pity_w) if w_ups > 0 else 0.0
    return e_char + e_weapon


def exact_combined_probability(n_pulls, c_ups, w_ups, guar_c, pity_c, pity_w):
    """精确计算在总预算 n_pulls 抽内抽齐 c_ups 角色 + w_ups 武器 的概率。
    通过两个独立分布的卷积计算：P(角色抽数 + 武器抽数 <= n_pulls)。"""
    if c_ups <= 0 and w_ups <= 0:
        return 1.0
    cdf_char = exact_cdf_k("character", n_pulls, c_ups, guar_c, pity_c) if c_ups > 0 \
        else np.ones(n_pulls + 1)
    cdf_weapon = exact_cdf_k("weapon", n_pulls, w_ups, False, pity_w) if w_ups > 0 \
        else np.ones(n_pulls + 1)
    # 武器恰好用 x 抽凑齐的概率
    pmf_weapon = np.diff(np.concatenate(([0.0], cdf_weapon)))
    total = 0.0
    for x in range(n_pulls + 1):
        total += pmf_weapon[x] * cdf_char[n_pulls - x]
    return float(total)


if __name__ == "__main__":
    print("这是抽卡模拟核心库。")
    print("  命令行模式请运行：python gacha_cli.py [参数]")
    print("  网页前端请运行  ：python app.py（或直接运行打包好的 鸣潮抽卡模拟器.exe）")
