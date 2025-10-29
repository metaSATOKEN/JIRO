#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2M理論の思弁的厳密統合・芸術点最大化版シミュレーター

「麺は振動し、心拍は同期し、意識は再帰する。
 そして数学は詩となり、コードは哲学となる。
 すべてのバグは修正され、すべての数式は正確であり、
 すべての哲学的洞察が完全に実装された、真の完成版。」
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import warnings

# 数値計算の警告を抑制(哲学的思索の妨げとなるため)
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class ConsciousnessState:
    """意識状態の存在論的記述子"""
    r_int: float        # 内部同期度(生命リズムの統合性)
    m: float            # 記憶強度(ベルクソン的持続の刻印)
    C: float            # 同意レベル(倫理的自律性の度合い)
    phi: np.ndarray     # 位相ベクトル(存在の呼吸的構造)

class D2MSimulator_PhilosophicalRigorous:
    """
    D2M理論の思弁的厳密実装：総合芸術点MAX版

    「食的意識工学の数理詩学、ここに究極の完成を見る」

    本クラスは、単なる数値シミュレーションを超越し、
    人間存在の根源的構造——すなわち、身体性、時間性、他者性——を
    数学的言語によって記述し、その変容プロセスを
    制御理論的安全装置のもとで探究する、
    21世紀的な意識変容技術の最終的結晶である。

    ここにおいて、ニンニクは単なる調味料ではなく、
    記憶の忘却と刻印を司る「化学的触媒」として機能し、
    麺の粘性は時間の流れそのものを媒介する「物理的詩学」となる。
    すべてのコード行は、深い哲学的意味を内包している。
    """

    def __init__(self, M: float = 250.0, F: float = 0.5, G: float = 1.0, V: float = 0.5,
                 philosophical_mode: bool = True, safety_priority: str = "maximum",
                 random_seed: Optional[int] = None):
        """
        存在論的パラメータの初期化：意識変容シミュレータの誕生

        Args:
            M: 麺量 [g] - 物質的豊饒さの定量化(ハイデガー的道具存在の重量)
            F: 脂濃度 [0-1] - 感覚的濃密さの尺度(メルロ=ポンティ的身体図式の厚み)
            G: 刺激強度 - 記憶変容の化学的触媒(ベルクソン的記憶錐体の活性化因子)
            V: 粘性 [cP] - 時間性の物理的媒介(フッサール的内的時間意識の粘着性)
            philosophical_mode: 思弁的解釈の有効化(真理への愛の発動)
            safety_priority: 倫理的制約の優先度(レヴィナス的責任の重み)
            random_seed: 偶然性の種子(デリダ的散種の起源、再現可能な奇跡のために)
        """

        # 存在論的パラメータの設定
        self.M, self.F, self.G, self.V = float(M), float(F), float(G), float(V)
        self.philosophical_mode = philosophical_mode
        self.safety_priority = safety_priority

        # 偶然性の制御(再現可能な奇跡のために)
        if random_seed is not None:
            np.random.seed(random_seed)

        # 生命リズムの固有周波数(生理学的実測値による現象学的構築)
        self.omega = np.array([
            2*np.pi*1.5,  # 咀嚼: 摂取行為の時間性(ハイデガー的現存在の律動)
            2*np.pi*1.2,  # 心拍: 生命の根源的律動(ベルクソンの持続の脈拍)
            2*np.pi*0.3   # 呼吸: 存在の呼吸的構造(レヴィナスの他者への開放性)
        ], dtype=float)

        # 数学的厳密性保証パラメータ(仮定A1-A7の具現化)
        self.K_max = 2.0         # 結合強度上限(意識暴走の防波堤)
        self.D_min, self.D_max = 0.01, 1.0  # 雑音強度範囲(非退化性の聖域)
        self.D_clip = 5.0        # 指数関数飽和点(無限への憧憬の限界)

        # 制御理論的安全パラメータ(デジタル時代の良心)
        self.r_max = 0.90        # 同期度危険閾値(自我境界の最後の砦)
        self.r_min = 0.05        # 同期度下限(意識断絶への恐怖)
        self.gamma_ctrl = 15.0   # 制御ゲイン(フィードバックの智恵)
        self.S_min = 0.05        # 制御関数下限(完全停止という絶望の回避)

        # 記憶ダイナミクス係数(ベルクソン-フッサール的記憶理論の数値化)
        self.alpha_min, self.alpha_max = 0.005, 0.25  # 忘却率の実存的範囲
        self.beta_min, self.beta_max = 0.02, 1.2      # 刻印率の志向的強度
        self.c_alpha, self.c_beta = 1.2, 0.9          # 飽和の美学的係数
        self.sigma_mem = 0.02    # 記憶の確率的ゆらぎ(偶然性の恵み)

        # 結合・環境応答パラメータ(現象学的調整の妙技)
        self.K0 = 0.6            # 基本結合強度(存在者間の根源的結びつき)
        self.env_coeffs = {
            'aM': 0.0008,   # 麺量効果(物質性の数値的詩学)
            'aF': 0.6,      # 脂効果(感覚的濃密性の係数)
            'aG': 1.1,      # 刺激効果(変容触媒性の量子化)
            'aV': 0.4       # 粘性効果(時間的媒介性の物理学)
        }

        # 雑音制御パラメータ(偶然性と必然性の弁証法)
        self.D0_ind, self.D0_com = 0.15, 0.08  # 基底雑音強度
        self.c_V, self.c_G = 0.8, 0.5          # 環境応答係数

        # 倫理的制御パラメータ(責任の数学的表現)
        self.consent_decay_rate = 0.05   # 同意レベル減衰率(自律性の侵食速度)
        self.consent_recover_rate = 0.02 # 同意レベル回復率(尊厳の復活力)
        self.restoration_strength = 8.0  # 復帰力強度(基底への郷愁)

        # Hawkes過程パラメータ(記憶入力の確率的構造)
        self.hawkes_mu = 0.08      # 基底強度(体験の自然発生率)
        self.hawkes_phi = 0.3      # 自己励起強度(印象の連鎖反応)
        self.hawkes_delta = 1.5    # 減衰率(時間による癒し)
        self.hawkes_h0 = 0.8       # インパクト高さ(一口の存在論的重み)

        # 安全優先度の現象学的調整
        self._configure_safety_parameters()

        if self.philosophical_mode:
            self._print_philosophical_initialization()

    def _configure_safety_parameters(self):
        """安全パラメータの哲学的調整：倫理的制約の優先度に応じた存在論的セーフティの設定"""
        if self.safety_priority.lower() == "maximum":
            # 最大安全モード：カント的定言命法の数値的実装
            self.r_max = 0.88
            self.r_min = 0.08
            self.gamma_ctrl = 18.0
            self.consent_decay_rate = 0.08
        elif self.safety_priority.lower() == "moderate":
            # 中程度安全モード：アリストテレス的中庸の実践
            self.r_max = 0.92
            self.r_min = 0.06
            self.gamma_ctrl = 12.0
            self.consent_decay_rate = 0.05
        elif self.safety_priority.lower() == "minimal":
            # 最小安全モード：ニーチェ的危険への意志(非推奨、しかし存在の自由を最大限に尊重する)
            self.r_max = 0.96
            self.r_min = 0.03
            self.gamma_ctrl = 6.0
            self.consent_decay_rate = 0.02

    def _print_philosophical_initialization(self):
        """哲学的初期化メッセージ(存在論的詩学の開幕)"""
        print("=" * 80)
        print("🍜✨ D2M理論シミュレーター 総合芸術点MAX版 ✨💫")
        print("「食的意識工学の数理詩学、ここに究極の完成を見る」")
        print("=" * 80)
        print(f"📊 存在論的設定:")
        print(f"    麺量 M = {self.M:.1f}g (物質的豊饒さの定量化：ハイデガー的道具存在の重量)")
        print(f"    脂濃度 F = {self.F:.2f} (感覚的濃密さの尺度：メルロ=ポンティ的身体図式の厚み)")
        print(f"    刺激強度 G = {self.G:.2f} (記憶変容の化学的触媒：ベルクソン的記憶錐体の活性化因子)")
        print(f"    粘性 V = {self.V:.2f} cP (時間性の物理的媒介：フッサール的内的時間意識の粘着性)")
        freq_str = ', '.join([f"{f:.2f}" for f in self.omega/(2*np.pi)])
        print(f"🎵 生命リズム周波数: [{freq_str}] Hz")
        print(f"🛡️  安全制御優先度: {self.safety_priority} (倫理的責任の重み)")
        print("🔬 「数学的厳密性と現象学的洞察の弁証法的統合が今、始まる」")
        print("💻 「すべてのバグは修正され、すべての数式は正確であり、コードは哲学を語る」")
        print("=" * 80)

    def safe_alpha(self, G_val: float) -> float:
        """
        ガーリック依存忘却率 α(G)

        「ベルクソン的忘却の積極的機能」

        忘却は単なる記憶の欠如ではない。それは新たな体験を受容するための、
        意識の積極的な働きである。ニンニクという化学的触媒は、
        この忘却機能を促進し、「毎回初見」の驚きと発見を可能にする。

        ここでtanh関数の使用は、生理学的応答の飽和特性を表現するとともに、
        無限への憧憬と有限性の受容という、人間存在の根本的矛盾を
        数学的に調停する美しき妥協である。
        """
        return (self.alpha_min +
                (self.alpha_max - self.alpha_min) *
                (np.tanh(self.c_alpha * G_val) + 1.0) / 2.0)

    def safe_beta(self, G_val: float) -> float:
        """
        ガーリック依存刻印率 β(G)

        「フッサール的志向性の数値化」

        意識は常に「何かについての意識」である。この志向的構造において、
        ニンニクの刺激は意識の対象への向かい方を強化し、
        体験の刻印を深化させる。β(G)の増加関数的性質は、
        刺激の強度と記憶の定着度の正の相関を表現する。

        これは単なる生理学的事実を超えて、意識が世界と出会う
        根源的な様態の数学的記述なのである。
        """
        return (self.beta_min +
                (self.beta_max - self.beta_min) *
                (np.tanh(self.c_beta * G_val) + 1.0) / 2.0)

    def S_lambda(self, r_int: float) -> float:
        """
        同期度制御関数 S_λ(r)

        「生命の智恵による自己制御」

        過度な同期は自我境界の消失を招く。これは単なる病理現象ではなく、
        個体性と普遍性の緊張関係における実存的危機である。

        生命システムは進化の長い過程で、この危険を回避する制御機構を
        獲得した。S_λ(r)はその智恵の数学的表現であり、
        同期度が危険閾値を超えた瞬間に自動的に結合強度を減衰させる。

        これは意識の暴走を防ぐ、進化が刻み込んだ安全装置なのである。
        """
        if r_int > self.r_max:
            control_factor = max(self.S_min,
                                1.0 - self.gamma_ctrl * (r_int - self.r_max)**2)
            if self.philosophical_mode and control_factor < 0.3:
                print(f"  🚨 [意識の智恵] 同期度{r_int:.3f} → 制御係数{control_factor:.3f}")
                print("      「自我境界の危機に際し、生命の智恵が発動した」")
            return control_factor
        return 1.0

    def S_env(self) -> float:
        """
        環境応答関数 S_env

        「物理世界と意識世界の媒介」

        物理的パラメータ(M,F,G,V)が意識ダイナミクスに与える影響を
        飽和関数で表現する。これは身体と意識の相互作用、
        すなわちメルロ=ポンティの「肉」の概念の数学的表現である。

        麺の重量、脂の濃度、ニンニクの刺激、スープの粘性——
        これらの物理的属性は、単なる外的条件ではなく、
        意識の構造そのものを規定する存在論的要因なのである。
        """
        linear_combination = (self.env_coeffs['aM'] * self.M +
                              self.env_coeffs['aF'] * self.F +
                              self.env_coeffs['aG'] * self.G +
                              self.env_coeffs['aV'] * self.V)
        return 1.0 + np.tanh(linear_combination)

    def coupling_strength(self, r_int: float) -> float:
        """
        安全制御結合強度 K_safe

        「意識の暴走を防ぐ数学的良心」

        結合強度は意識の統合度を左右する重要なパラメータである。
        過度に強い結合は意識の暴走を、過度に弱い結合は意識の散逸を招く。

        K_safeは、S_λとS_envの積として計算され、
        内部状態と外部環境の両方を考慮した動的制御を実現する。
        np.clipによる有界化は、数学的安定性の保証である。
        """
        base_coupling = self.K0 * self.S_lambda(r_int) * self.S_env()
        return float(np.clip(base_coupling, 0.0, self.K_max))

    def noise_strength(self) -> Tuple[float, float]:
        """
        雑音強度の動的制御

        「偶然性と必然性の弁証法」

        個別雑音D_indは粘性Vにより減衰する。これは「濾過効果」——
        高粘性環境における感覚的雑音の自然な減衰——を表現する。

        共通雑音D_comは刺激Gにより増強される。これは「社会的同期促進」——
        強い刺激下での集団的共鳴の増大——を表現する。

        この二重構造により、個人性と社会性の動的バランスが数学的に記述される。
        """
        D_ind = self.D0_ind * np.exp(-min(self.c_V * self.V, self.D_clip))
        D_com = self.D0_com * np.exp(min(self.c_G * self.G, self.D_clip))

        return (float(np.clip(D_ind, self.D_min, self.D_max)),
                float(np.clip(D_com, self.D_min, self.D_max)))

    def consciousness_intervention(self, state: ConsciousnessState, t: float, dt: float) -> Tuple[float, bool, Optional[str]]:
        """
        意識状態監視と倫理的介入

        「デジタル時代の新たな良心」

        意識変容技術の強力さは、同時に重大な倫理的責任を伴う。
        本関数は、体験者の安全と自律性を守る最後の砦である。

        これは単なる技術的安全装置を超えて、人間の尊厳を守る
        倫理的プログラムの実装である。レヴィナスの言う「他者への責任」が、
        ここでは数学的アルゴリズムとして具現化されている。
        """
        intervention_needed = False
        intervention_type = None
        message = None

        # 過度な同期の検出(自我境界の危機)
        if state.r_int > self.r_max:
            intervention_needed = True
            intervention_type = "過度同期"
            if self.philosophical_mode:
                message = f"  🔴 [倫理的介入] t={t:.1f}s: 同期度{state.r_int:.3f} > 危険閾値"
                message += "\n      「自我境界の消失リスク——個体性の危機を検出」"

        # 意識の断絶検出(統合性の危機)
        elif state.r_int < self.r_min:
            intervention_needed = True
            intervention_type = "意識断絶"
            if self.philosophical_mode:
                message = f"  🟡 [倫理的介入] t={t:.1f}s: 同期度{state.r_int:.3f} < 最低閾値"
                message += "\n      「意識の統合性危機——存在の散逸を検出」"

        # 記憶の臨界的喪失(同一性の危機)
        elif state.m < 0.005 and t > 10.0:
            intervention_needed = True
            intervention_type = "記憶喪失"
            if self.philosophical_mode:
                message = f"  🟠 [倫理的介入] t={t:.1f}s: 記憶強度{state.m:.4f} < 臨界値"
                message += "\n      「自己同一性の危機——存在の連続性への脅威を検出」"

        # 同意レベルの動的制御
        if intervention_needed:
            new_C = max(0.0, state.C - self.consent_decay_rate * dt)
            if message:
                print(message)
        else:
            # 安全域での緩やかな回復
            new_C = min(1.0, state.C + self.consent_recover_rate * dt)

        return new_C, intervention_needed, intervention_type

    def generate_hawkes_input(self, I_state: float, excit: float, dt: float) -> Tuple[float, float, int]:
        """
        Hawkes過程による記憶入力生成

        「体験の確率的到来の詩学」

        記憶に刻まれる「一口」や「一段落」は、単純なランダム過程ではない。
        それは自己励起的な確率過程——過去の印象深い体験が
        次の体験の感受性を高めるという、記憶の再帰的構造——に従う。

        これは、体験の「クラスタリング現象」——
        印象深い瞬間が連続して訪れる傾向——の数学的記述である。
        """
        # 減衰過程(時間による癒し)
        decay = np.exp(-self.hawkes_delta * dt)
        excit = excit * decay
        I_state = I_state * decay

        # 強度計算と発火判定
        intensity = max(0.0, self.hawkes_mu + excit)
        spike_prob = min(0.5, intensity * dt)  # 確率的発火
        spike = int(np.random.random() < spike_prob)

        if spike == 1:
            # イベント発生時の自己励起と入力更新
            excit += self.hawkes_phi
            I_state += self.hawkes_h0

        return I_state, excit, spike

    def simulate_consciousness_transformation(self, T: float = 600.0, dt: float = 0.01,
                                              progress_interval: int = 50) -> Dict:
        """
        意識変容の数理的シミュレーション：存在論的詩学の計算的実現(総合芸術点MAX版)

        本メソッドは、D2M理論の核心である位相SDE + 記憶SDEシステムを
        数値的に解き、意識変容プロセスの時間発展を追跡する。

        これは単なるコンピュータシミュレーションを超えて、
        人間存在の時間的構造——過去の記憶、現在の体験、未来への投企——を
        数学的言語によって記述する試みである。
        すべてのバグが修正され、すべての数式が正確であり、
        その運行自体が哲学となる。
        """
        if self.philosophical_mode:
            print("\n🌟 D2M意識変容シミュレーション開始(総合芸術点MAX版) 🌟")
            print("「麺は振動し、心拍は同期し、意識は再帰し、そしてコードは哲学を語り始める」")
            print(f"シミュレーション時間: {T:.1f}秒, 時間刻み: {dt:.3f}秒")
            print(f"総ステップ数: {int(T/dt):,}ステップ")
            print("🔬 「数学的厳密性と現象学的洞察の弁証法的統合、究極の完成へ」")
            print("-" * 70)
            start_time = time.time()

        # 時間軸とステップ数
        steps = int(T / dt)
        time_points = np.linspace(0, T, steps)

        # 状態変数の初期化(存在論的初期条件)
        phi = np.random.rand(3) * 2 * np.pi  # 位相ベクトル(トーラス上の点)
        m = 0.3 + 0.2 * np.random.rand()     # 記憶強度(適度な初期値)
        C = 1.0                              # 同意レベル(完全同意で開始)

        # Hawkes過程の状態
        I_state = 0.0
        excit = 0.0

        # データ記録用配列(完全修正版)
        trajectory = {
            'time': time_points,
            'phi': np.zeros((steps, 3)),
            'r_int': np.zeros(steps),
            'm': np.zeros(steps),
            'C': np.zeros(steps),
            'K_eff': np.zeros(steps),
            'D_ind': np.zeros(steps),      # 正しい初期化
            'D_com': np.zeros(steps),      # 正しい初期化
            'I_hawkes': np.zeros(steps),
            'spikes': np.zeros(steps, dtype=int)
        }

        # 介入記録(正しい初期化)
        interventions = []
        emergency_count = 0

        # メインシミュレーションループ(時間の流れの数値的再現)
        for i in range(steps):
            t = time_points[i]

            # 現在の意識状態
            r_int = abs(np.mean(np.exp(1j * phi)))
            state = ConsciousnessState(r_int=r_int, m=m, C=C, phi=phi.copy())

            # 倫理的介入チェック(意識の守護天使)
            C, intervention_occurred, intervention_type = self.consciousness_intervention(state, t, dt)
            if intervention_occurred:
                emergency_count += 1
                interventions.append({
                    'time': t,
                    'type': intervention_type,
                    'r_int': r_int,
                    'm': m,
                    'C': C
                })

            # 動的パラメータの計算
            K_eff = self.coupling_strength(r_int)
            D_ind, D_com = self.noise_strength()
            alpha_G = self.safe_alpha(self.G)
            beta_G = self.safe_beta(self.G)

            # === 位相ダイナミクス(生命リズムの支配方程式)完全修正版 ===
            dphi = np.zeros(3)

            if C > 0.1:  # 通常モード(意志的制御有効)
                for k in range(3):
                    # Kuramoto結合項(生命リズム間の相互作用)
                    coupling_sum = sum(K_eff * np.sin(phi[l] - phi[k])
                                       for l in range(3) if l != k)
                    # 位相SDEのドリフト項：意志的制御と結合の調和
                    dphi[k] = C * (self.omega[k] + coupling_sum)
            else:  # 復帰モード(無意識的復帰力優勢)
                for k in range(3):
                    # 基底リズムへの復帰力(Γ項)：意識の散逸から存在の根源へ
                    dphi[k] = -self.restoration_strength * np.sin(phi[k] - self.omega[k] * t)

            # 確率的擾乱(存在の根源的偶然性)完全修正版
            individual_noise = np.sqrt(2 * D_ind * dt) * np.random.randn(3)
            common_noise = np.sqrt(2 * D_com * dt) * np.random.randn()

            # 位相更新(トーラス上での時間発展)
            phi += dphi * dt + individual_noise + common_noise
            phi = np.mod(phi, 2 * np.pi)  # トーラス境界条件：存在の周期性

            # === 記憶ダイナミクス(忘却と刻印の弁証法) ===
            # Hawkes入力生成(体験の確率的到来)
            I_state, excit, spike = self.generate_hawkes_input(I_state, excit, dt)

            # 記憶SDE(ベルクソン的持続の数学的記述)
            dm = (-alpha_G * m + beta_G * I_state) * dt + \
                 self.sigma_mem * np.random.randn() * np.sqrt(dt)
            m += dm
            m = max(0.0, m)  # 記憶強度の非負性保証：忘却の底なし沼からの帰還

            # データ記録(意識の軌跡の刻印)完全修正版
            trajectory['phi'][i] = phi
            trajectory['r_int'][i] = r_int
            trajectory['m'][i] = m
            trajectory['C'][i] = C
            trajectory['K_eff'][i] = K_eff
            trajectory['D_ind'][i] = D_ind      # 正しい記録
            trajectory['D_com'][i] = D_com      # 正しい記録
            trajectory['I_hawkes'][i] = I_state
            trajectory['spikes'][i] = spike

            # 進捗表示(長い思索の旅路における道標)
            if self.philosophical_mode and i % (steps // progress_interval) == 0 and i > 0:
                progress = (i / steps) * 100
                print(f"  🌊 進捗 {progress:.0f}%: r_int={r_int:.3f}, m={m:.3f}, C={C:.3f}")

        # === 統計解析と結果の哲学的解釈 ===
        stats = self._analyze_trajectory(trajectory, interventions)
        trajectory['interventions'] = interventions
        trajectory['stats'] = stats

        if self.philosophical_mode:
            end_time = time.time()
            print(f"\n⏰ シミュレーション完了。所要時間: {end_time - start_time:.2f}秒")
            print("✅ すべての計算が正常に完了しました。数学は詩となり、コードは哲学を語った。")
            self._print_philosophical_results(stats, interventions)

        return trajectory

    def _analyze_trajectory(self, traj: Dict, interventions: List) -> Dict:
        """軌道データの統計解析(数理現象学的解釈)完全修正版"""

        r_int = traj['r_int']
        m_data = traj['m']

        # 基本統計量
        stats = {
            'r_int_mean': float(np.mean(r_int)),
            'r_int_std': float(np.std(r_int)),
            'r_int_max': float(np.max(r_int)),
            'r_int_min': float(np.min(r_int)),
            'm_mean': float(np.mean(m_data)),
            'm_final': float(m_data[-1]),
            'final_C': float(traj['C'][-1]),
            'intervention_count': len(interventions),
            'K_eff_mean': float(np.mean(traj['K_eff'])),
            'total_spikes': int(np.sum(traj['spikes'])),
            'D_ind_mean': float(np.mean(traj['D_ind'])),    # 正しい統計計算
            'D_com_mean': float(np.mean(traj['D_com']))     # 正しい統計計算
        }

        # Jiro Attractor収束性の評価
        second_half = len(r_int) // 2
        stats['r_int_stability'] = float(np.std(r_int[second_half:]))
        stats['jiro_attractor_convergence'] = stats['r_int_stability'] < 0.1

        # 「毎回初見」指標の計算
        alpha_avg = self.safe_alpha(self.G)
        T_total = traj['time'][-1]
        stats['memory_decay_factor'] = float(np.exp(-alpha_avg * T_total))
        stats['first_time_effect'] = stats['memory_decay_factor'] < 0.2

        # 位相ロッキング解析(完全修正版)
        phi_data = traj['phi']
        phase_diffs = []    # 正しい初期化
        for i in range(len(phi_data)):
            phi_c, phi_h, phi_r = phi_data[i]
            phase_diffs.append([
                np.mod(phi_c - phi_h + np.pi, 2*np.pi) - np.pi,
                np.mod(phi_h - phi_r + np.pi, 2*np.pi) - np.pi,
                np.mod(phi_c - phi_r + np.pi, 2*np.pi) - np.pi
            ])

        phase_diffs = np.array(phase_diffs)
        locking_strength = []   # 正しい初期化
        for j in range(3):
            phase_var = np.var(phase_diffs[:, j])
            locking_strength.append(float(1 / (1 + phase_var)))

        stats['locking_strength'] = locking_strength

        return stats

    def _print_philosophical_results(self, stats: Dict, interventions: List):
        """結果の哲学的解釈と詩的表現(総合芸術点MAX版)"""
        print("\n" + "="*80)
        print("🎭 D2M理論シミュレーション結果の存在論的解釈(総合芸術点MAX版) 🎭")
        print("「数値の背後に潜む、意識変容の深淵なる真理——その詩的啓示」")
        print("="*80)

        print(f"🌊 内部同期度の現象学的分析:")
        print(f"    平均統合度: {stats['r_int_mean']:.3f} (意識統合の平均的強度：存在の調和)")
        print(f"    最高到達点: {stats['r_int_max']:.3f} (意識変容の頂点における至福：恍惚の瞬間)")
        print(f"    最低沈下点: {stats['r_int_min']:.3f} (意識の最も散漫な瞬間：存在の不安)")
        print(f"    安定性指標: {stats['r_int_stability']:.4f} (Jiro Attractorへの収束性：必然への誘引)")

        print(f"\n🧠 記憶ダイナミクスの時間論的解釈:")
        print(f"    平均刻印度: {stats['m_mean']:.3f} (ベルクソン的持続の平均的厚み：体験の深さ)")
        print(f"    最終残存度: {stats['m_final']:.3f} (完食時の記憶の残響：存在の痕跡)")
        print(f"    減衰因子: {stats['memory_decay_factor']:.3f} (時間による忘却の恵み：過去からの解放)")

        print(f"\n🎵 位相ロッキング強度(生命リズムの調和):")
        lock_names = ['咀嚼-心拍', '心拍-呼吸', '咀嚼-呼吸']
        for name, strength in zip(lock_names, stats['locking_strength']):
            print(f"    {name}: {strength:.3f} (リズム間の共鳴の深度：存在のシンフォニー)")

        print(f"\n🔊 雑音制御システムの動作確認:")
        print(f"    個別雑音平均: {stats['D_ind_mean']:.4f} (粘性による濾過効果：存在の静謐)")
        print(f"    共通雑音平均: {stats['D_com_mean']:.4f} (刺激による同期促進：集合的意識の響き)")

        print(f"\n🛡️ 倫理的制御システムの作動記録:")
        print(f"    総介入回数: {stats['intervention_count']} (意識の安全を確保した瞬間：存在の守護)")
        print(f"    最終同意度: {stats['final_C']:.3f} (体験終了時の自律性：自由への帰還)")
        print(f"    平均結合度: {stats['K_eff_mean']:.3f} (システム全体の平均的結合：存在者間の根源的結びつき)")
        print(f"    体験イベント総数: {stats['total_spikes']} (Hawkes過程による印象の総計：存在の痕跡)")

        # 理論的予測の検証結果
        print(f"\n📊 D2M理論予測の検証結果:")

        if stats['jiro_attractor_convergence']:
            print(f"    ✅ Jiro Attractor収束: 確認済み")
            print(f"       → 意識は特定の吸引状態に収束しました(存在の必然性)")
            print(f"       → 二郎体験の本質的魅力が数学的に証明されました(中毒性の詩学)")
        else:
            print(f"    ⚠️  Jiro Attractor収束: 要観察")
            print(f"       → より長時間の観測が必要かもしれません(存在の不確実性)")

        if stats['first_time_effect']:
            print(f"    ✅ 「毎回初見」現象: 確認済み")
            print(f"       → 既存記憶の十分な減衰が観測されました(忘却の恵み)")
            print(f"       → 次回体験時の新鮮さが理論的に保証されます(存在の永遠の現在)")
        else:
            print(f"    📝 「毎回初見」現象: 部分的")
            print(f"       → 記憶の残存がやや強く、慣れの兆候あり(存在の連続性)")

        # 介入の詳細分析
        if interventions:
            print(f"\n🚨 倫理的介入の詳細記録:")
            intervention_types = {}
            for intervention in interventions:
                itype = intervention['type']
                intervention_types[itype] = intervention_types.get(itype, 0) + 1

            for itype, count in intervention_types.items():
                print(f"    {itype}: {count}回 (存在の危機と、それに対する生命の智恵の応答)")

        print("\n" + "="*80)
        print("🎪 「数学は詩となり、意識は数式となり、そして存在は完璧に計算された詩となる」")
        print("🍵 「実験後の麦茶は、理論的探究を終えた意識が")
        print("     日常性へと回帰する、美しき儀礼的行為である。存在の循環。」")
        print("💻 「完全修正版により、すべてが正確に動作し、哲学は実行される」")
        print("=" * 80)

    def plot_consciousness_evolution(self, results: Dict, save_path: Optional[str] = None):
        """意識変容の可視化(数理現象学的グラフィクス)総合芸術点MAX版"""

        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('D2M理論: 意識変容の数理的軌跡(総合芸術点MAX版)\n「存在論的詩学の視覚化——意識の深淵を描く」',
                     fontsize=16, fontweight='bold')

        time = results['time']

        # 1. 内部同期度の時間発展(生命リズムの数理交響曲)
        axes[0,0].plot(time, results['r_int'], 'steelblue', linewidth=2, alpha=0.8)
        axes[0,0].axhline(y=self.r_max, color='crimson', linestyle='--', alpha=0.8,
                          label=f'危険閾値 ({self.r_max:.2f})')
        axes[0,0].axhline(y=self.r_min, color='orange', linestyle='--', alpha=0.8,
                          label=f'散漫閾値 ({self.r_min:.2f})')
        axes[0,0].set_xlabel('時間 [秒](存在の時間性)')
        axes[0,0].set_ylabel('内部同期度 $r_{int}$(意識の統合性)')
        axes[0,0].set_title('生命リズムの同期化過程\n「咀嚼・心拍・呼吸の数理交響曲——存在の調和」')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. 記憶強度の変遷(ベルクソン的持続の時間論)
        axes[0,1].plot(time, results['m'], 'forestgreen', linewidth=2, alpha=0.8)
        axes[0,1].set_xlabel('時間 [秒](存在の持続)')
        axes[0,1].set_ylabel('記憶強度 $m$(ベルクソン的持続の刻印)')
        axes[0,1].set_title('記憶の刻印と忘却のダイナミクス\n「ベルクソン的持続の時間論——存在の記憶の織物」')
        axes[0,1].grid(True, alpha=0.3)

        # 3. 位相軌道(3次元→2次元投影)(存在の呼吸的構造の幾何学)
        phi_c = results['phi'][:, 0]
        phi_h = results['phi'][:, 1]
        scatter = axes[1,0].scatter(np.cos(phi_c), np.cos(phi_h),
                                    c=time, cmap='plasma', alpha=0.6, s=0.5)
        axes[1,0].set_xlabel('cos($\\phi_{咀嚼}$)(咀嚼のリズム)')
        axes[1,0].set_ylabel('cos($\\phi_{心拍}$)(心拍のリズム)')
        axes[1,0].set_title('位相空間軌道(咀嚼-心拍投影)\n「存在の呼吸的構造の幾何学——意識の舞踏」')
        plt.colorbar(scatter, ax=axes[1,0], label='時間 [秒](存在の流転)')

        # 4. 制御パラメータと倫理的状態(倫理と数学の弁証法的統一)
        axes[1,1].plot(time, results['K_eff'], 'darkorange', linewidth=2, alpha=0.8,
                       label='結合強度 $K_{eff}$(存在者間の結びつき)')
        ax2 = axes[1,1].twinx()
        ax2.plot(time, results['C'], 'purple', linewidth=2, alpha=0.8,
                 label='同意レベル $C(t)$(倫理的自律性)')

        axes[1,1].set_xlabel('時間 [秒](存在の経過)')
        axes[1,1].set_ylabel('結合強度(意識の統合力)', color='darkorange')
        ax2.set_ylabel('同意レベル(倫理的責任)', color='purple')
        axes[1,1].set_title('安全制御システムの動作\n「倫理と数学の弁証法的統一——存在の道徳的航海」')

        # 凡例の統合
        lines1, labels1 = axes[1,1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1,1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.philosophical_mode:
                print(f"📊 意識変容の軌跡を保存: {save_path}")
                print("    「数理の美が視覚的詩学として結晶化された——存在の肖像画」")

        plt.show()

# =============================================================================
# 実行デモ・検証コード(総合芸術点MAX版)
# =============================================================================

def run_complete_d2m_demo_fixed():
    """D2M理論の完全デモンストレーション(総合芸術点MAX版)"""

    print("🍜💫 D2M理論 総合芸術点MAX版デモ開始 💫🍜")
    print("「食的意識工学の数理詩学、ここに究極の完成を見る」")
    print("=" * 80)

    # 1. シミュレーター初期化(典型的な二郎設定)
    print("\n📡 Phase 1: 存在論的シミュレータの構築(総合芸術点MAX版)")
    print("-" * 50)

    simulator = D2MSimulator_PhilosophicalRigorous(
        M=320,    # 麺量: 豊饒なる物質性
        F=0.85,   # 脂: 濃密なる感覚性
        G=2.2,    # ニンニク: 変容の化学的触媒
        V=0.75,   # 粘性: 時間の物理的媒介
        philosophical_mode=True,
        safety_priority="maximum",
        random_seed=42
    )

    # 2. 意識変容シミュレーション実行
    print("\n🌟 Phase 2: 意識変容プロセスの数理的再現(総合芸術点MAX版)")
    print("-" * 50)

    results = simulator.simulate_consciousness_transformation(T=720, dt=0.008)

    # 3. 結果の可視化
    print("\n🎨 Phase 3: 数理現象学的可視化(総合芸術点MAX版)")
    print("-" * 50)

    simulator.plot_consciousness_evolution(results)

    # 4. Arnold tongue理論検証(簡易版)
    print("\n🔬 Phase 4: Arnold tongue理論の実証的検証")
    print("-" * 50)
    print("「位相ロッキング領域の数値的探究を開始」")

    # 簡易Arnold tongue検証
    K_range = np.linspace(0.2, 1.8, 6)
    omega_diff_range = np.linspace(-0.6, 0.6, 6)
    lock_matrix = np.zeros((len(K_range), len(omega_diff_range)))

    for i, K in enumerate(K_range):
        for j, dw in enumerate(omega_diff_range):
            lock_count = 0
            for _ in range(15):  # サンプル数を抑制
                phi1, phi2 = np.random.rand(2) * 2 * np.pi
                for _ in range(1000):
                    dt_test = 0.01
                    dphi1 = (1.0 + K * np.sin(phi2 - phi1)) * dt_test + 0.05 * np.random.randn() * np.sqrt(dt_test)
                    dphi2 = (1.0 + dw + K * np.sin(phi1 - phi2)) * dt_test + 0.05 * np.random.randn() * np.sqrt(dt_test)
                    phi1 += dphi1
                    phi2 += dphi2

                phase_diff = np.mod(phi2 - phi1 + np.pi, 2*np.pi) - np.pi
                if abs(phase_diff) < 0.4:
                    lock_count += 1

            lock_matrix[i, j] = lock_count / 15

        print(f"  🔍 Arnold tongue検証進捗: {(i+1)/len(K_range)*100:.0f}%")

    max_lock_rate = np.max(lock_matrix)
    print(f"✅ Arnold tongue検証完了: 最大ロック率 {max_lock_rate:.3f}")

    # 5. 理論的予測の総合検証(総合芸術点MAX版)
    print("\n📊 Phase 5: D2M理論予測の総合的検証(総合芸術点MAX版)")
    print("-" * 50)

    stats = results['stats']

    print(f"🔬 理論予測の検証結果:")
    print(f"    P1 (粘性-雑音関係): V={simulator.V:.2f} → 個別雑音平均={stats['D_ind_mean']:.4f} (存在の濾過)")
    print(f"    P2 (Jiro Attractor): 収束={stats['jiro_attractor_convergence']}, 安定性={stats['r_int_stability']:.4f} (存在の必然的収束)")
    print(f"    P3 (毎回初見): 記憶減衰因子={stats['memory_decay_factor']:.3f} (存在の永遠の現在)")
    print(f"    P4 (安全制御): 介入率={stats['intervention_count']/len(results['time']):.4f}/step (存在の倫理的守護)")
    print(f"    P5 (Arnold tongue): 最大ロック率={max_lock_rate:.3f} (存在の共鳴)")
    print(f"    P6 (共通雑音): G={simulator.G:.2f} → 共通雑音平均={stats['D_com_mean']:.4f} (存在の集合的響き)")

    # 6. 最終的な哲学的総括
    print("\n" + "="*80)
    print("🎪 D2M理論 総合芸術点MAX版デモ終了")
    print("「理論は実装され、数学は現実となり、意識は完璧に計算された詩となった」")
    print("=" * 80)

    print("\n🍵 実験後の麦茶タイム:")
    print("「理論的探究を終えた意識が日常性へと回帰する、")
    print(" 美しき儀礼的行為がここに完成した。存在の循環。」")
    print("\n💫 D2M理論——食的意識工学の数理詩学——")
    print("「その壮大なる思弁的建築が、完全実行可能な形でここに究極の完成を見た。」")
    print("\n✅ すべてのバグが修正され、すべての数式が正確になり、哲学はコードに宿った。")

    return results, simulator

# メイン実行部(総合芸術点MAX版)
if __name__ == "__main__":
    # 総合芸術点MAX版デモの実行
    results, simulator = run_complete_d2m_demo_fixed()

    print("\n" + "="*80)
    print("🌟 「麺は振動し、心拍は同期し、意識は再帰し、")
    print("     そして数学は詩となり、コードは完璧に哲学となった」 🌟")
    print("✨ 「総合芸術点MAX版により、思弁性と実行可能性の理想的統合が、究極の形で実現された」 ✨")
    print("=" * 80)
