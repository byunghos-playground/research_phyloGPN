# research_phyloGPN

**PhyloGPN F81 Framework 검증 — Simulation Study**

Albors et al. 2025 (RECOMB) *A Phylogenetic Approach to Genomic Language Modeling* 에서 제안한
F81 phylogenetic loss framework를 **시뮬레이션 데이터**로 검증하는 프로젝트.

> 논문 저자들이 F81 loss 구현 코드를 공개하지 않아 논문 수식 기반으로 직접 구현.

---

## 세 가지 모델 (모든 실험 공통)

| 모델 | Loss | π_true 사용 | Alignment 사용 |
|------|------|-------------|----------------|
| **F81** | `-log P_F81(alignment \| π_pred, T) + log π_ref` | ✗ | ✓ |
| **F81 Supervised** | `log P_F81(alignment \| π_true, T) - log P_F81(alignment \| π_pred, T)` | ✓ (간접) | ✓ |
| **Naive** | `KL(π_true \|\| π_pred)` | ✓ (직접) | ✗ |

**핵심 질문**: alignment + tree의 evolutionary signal로 학습한 F81이 직접 지도 학습(Naive)에 비해 어떤가?

---

## 여섯 가지 실험

| 실험 | π 설계 | rate | branch_scale | 데이터셋 클래스 | shortcut |
|------|--------|------|--------------|----------------|----------|
| **exp1_baseline** | chunk당 하나, Dirichlet(1,1,1,1) | 없음 | 1.0 | `SimF81Dataset` | ✓ (있음) |
| **exp2_baseline_r** | chunk당 하나, Dirichlet(1,1,1,1) | per-site r ~ Gamma(0.5) | 1.0 | `SimF81Dataset` | 부분적 |
| **exp3_gc** | GC OU process, position-varying | 없음 | 1.0 | `WindowedSimF81Dataset` | ✗ |
| **exp4_gc_r** | GC OU process, position-varying | per-site r ~ Gamma(0.5) | 1.0 | `WindowedSimF81Dataset` | ✗ |
| **exp5_gc** | GC OU process, position-varying | 없음 | **10.0** | `WindowedSimF81Dataset` | ✗ |
| **exp6_gc_r** | GC OU process, position-varying | per-site r ~ Gamma(0.5) | **10.0** | `WindowedSimF81Dataset` | ✗ |

**branch_scale**: Zoonomia 계통수의 branch length를 일괄 곱함 (exp5/6은 ×10 → 더 많은 substitution → 더 강한 evolutionary signal).

---

## 시뮬레이션 상세

### Exp1 / Exp2 — chunk-level π (pyvolve 사용)

**방식**: 하나의 481bp 청크마다 π 하나를 샘플하고, 그 π로 전체 481 사이트를 시뮬레이션.

```
π ~ Dirichlet(α=1, α=1, α=1, α=1)   # symmetric = uniform over simplex
```

- `simulate_exp1_baseline.py`: rate=1 (constant)
- `simulate_exp2_baseline_r.py`: per-site rate r ~ Gamma(shape=0.5, scale=1.0)
- pyvolve로 F81 model + single Partition(size=481) → FASTA 생성
- pi.txt: L개 행 모두 동일한 π 값 (fasta_to_npz.py 포맷 호환)

**shortcut 원인**: 청크 전체(481bp)가 같은 π에서 생성 → ref_seq empirical frequency ≈ π
→ 모델이 단순 frequency counting으로 loss=0 달성 (epoch 3 이후).

---

### Exp3 / Exp4 / Exp5 / Exp6 — position-varying π (custom F81 simulation)

**pyvolve를 사용하지 않는 이유**: pyvolve는 per-position π를 지원하지 않음
(Partition 10,000개 생성 시 비현실적). numpy로 직접 F81 forward simulation 구현.

#### GC OU process (π 생성)

GC content를 **Ornstein-Uhlenbeck process**로 genome 전체에 걸쳐 시뮬레이션:

```
GC[0] ~ N(μ, σ²),  clip to [gc_min, gc_max]

Exact discrete update (step=1bp):
  GC[i] = GC[i-1] * exp(-θ) + μ * (1 - exp(-θ)) + ε
  ε ~ N(0, σ² * (1 - exp(-2θ)))
```

| 파라미터 | 값 | 의미 |
|----------|----|------|
| μ | 0.41 | OU mean GC content |
| σ | 0.07 | stationary std |
| θ | 1/5000 | mean-reversion rate |
| 상관 길이 | ~5,000 bp | GC content가 서서히 변함 |
| clip range | [0.30, 0.65] | Bernardi isochore family 커버 |

GC → π 변환:
```
π(x) = [(1-GC(x))/2,  GC(x)/2,  GC(x)/2,  (1-GC(x))/2]
         [     A     ,     C   ,     G   ,        T     ]
```

#### F81 forward simulation (top-down)

```python
# root: sample from π_x (stationary distribution)
# each child node:
#   P(child=j | parent=i, t) = π_j * (1 - exp(-t)) + δ_{ij} * exp(-t)
# where t = branch_length * branch_scale
```

preorder traversal로 root → leaf 방향으로 각 종의 서열 생성.
per-site rate variation (exp4, exp6): `t_eff = rate[x] * branch_length * branch_scale`.

#### shortcut 차단 원리

- π는 5,000bp 상관 길이로 천천히 변함
- 모델의 receptive field = 481bp (< 5,000bp)
- 481bp window 내 empirical frequency는 window 평균 π만 반영 → 정확한 center-site π 복원 불가
- → 모델이 evolutionary signal (MSA + tree)을 실제로 활용해야 함

---

## 아키텍처: RCEByteNet

| 항목 | 값 |
|------|----|
| outer_dim | 960 |
| inner_dim | 480 |
| Kernel size | 5 |
| Dilation 패턴 | [1, 5] × 20 = 40 blocks |
| Receptive field | 481 bp |
| 파라미터 수 | ~83M |

- **RCE**: DNA 양쪽 가닥 동등성을 weight parametrization으로 강제
- **Input**: ref_seq (L + 480 padding) → **Output**: π(4) per valid position
- **Exp1/2**: L_out = L = 481 (전체 positions에 loss 계산)
- **Exp3/4/5/6**: L_out = 1 (center position만 출력, valid conv RF=481)

### Felsenstein pruning (math_f81.py)

247종 Zoonomia 계통수에서 246번 확률 곱셈 → float32 underflow 방지:
- 각 내부 노드에서 amax 기반 per-node rescaling + log_scale_acc 누적
- GPU tensor 연산으로 벡터화

---

## 데이터 파이프라인

### Exp1 / Exp2

```
[Step 1] 시뮬레이션  (SLURM array)
  sbatch --array=0-999 scripts/exp{1,2}_*/simulate.sbatch
  → data/exp{1,2}_*/raw/chunk_NNNNN.fasta
  → data/exp{1,2}_*/raw/chunk_NNNNN_pi.txt
  → data/exp{1,2}_*/processed/block_NNNNN.npz   (fasta_to_npz.py 자동 실행)

[Step 2] 데이터 분할  (80/10/10)
  python data/simulate/split_data.py --data_dir data/exp1_baseline/processed ...
  → split.json 생성 (checkpoints/{exp}/{model}/split.json)
  ※ exp1은 사후 생성: python scripts/save_exp1_split.py

[Step 3] 훈련
  sbatch scripts/exp1_baseline/train_f81.sbatch          (→ checkpoints/exp1_baseline/f81/)
  sbatch scripts/exp1_baseline/train_f81_supervised.sbatch
  sbatch scripts/exp1_baseline/train_naive.sbatch
```

**npz 구조** (exp1/2):
```
ref_seq    : (481,)      int8  — reference sequence
pi_true    : (481, 4)    float32 — 모든 행 동일 (chunk당 π 하나)
msa_codes  : (481, S)    int8  — S=247 species alignment
taxon_names: (S,)        str
```

---

### Exp3 / Exp4 / Exp5 / Exp6

```
[Step 1] 시뮬레이션  (SLURM array, 100 jobs × 30 genomes)
  sbatch scripts/exp{3,4,5,6}_*/simulate.sbatch
  → data/exp{3,4}_*/raw/genome_NNNNN.fasta
  → data/exp{3,4}_*/raw/genome_NNNNN_pi.txt
  → data/exp{3,4,5,6}_*/processed/genome_NNNNN.npz   (fasta_to_npz.py 자동 실행)
  ※ exp4/6만 genome_NNNNN_r.txt 추가 출력 (per-site rates)

[Step 2] 훈련 (split.json은 train script가 자동 생성 — 80/10/10, seed=42)
  sbatch scripts/exp3_gc/train_f81_gc.sbatch          (→ checkpoints/exp3_gc/f81/)
  sbatch scripts/exp3_gc/train_f81_supervised_gc.sbatch
  sbatch scripts/exp3_gc/train_naive_gc.sbatch
```

**npz 구조** (exp3/4/5/6):
```
ref_seq    : (10000,)    int8  — reference sequence
pi_true    : (10000, 4)  float32 — position마다 다른 π
msa_codes  : (10000, S)  int8
taxon_names: (S,)        str
```

**WindowedSimF81Dataset**: genome npz를 stride=1 sliding window로 분해.
- center ∈ [240, 9760) — edge positions 제외 (padding 필요)
- 각 item: 481bp window, center_idx=240

---

## 평가

```bash
# exp1-6 한 번에 제출
for EXP in exp1_baseline exp2_baseline_r exp3_gc exp4_gc_r exp5_gc exp6_gc_r; do
    sbatch scripts/${EXP}/eval.sbatch
done
```

| 실험 | 평가 스크립트 | 데이터셋 클래스 |
|------|--------------|----------------|
| exp1, exp2 | `evaluate.py` | `SimF81Dataset` |
| exp3, exp4, exp5, exp6 | `evaluate_gc.py` | `WindowedSimF81Dataset` (stride=481) |

**평가 지표**:
- MAE: `|π_pred - π_true|` 평균 (각 염기별 + 전체)
- Pearson r: π_pred vs π_true 선형 상관 (각 염기별 + 전체 4N 쌍)
- KL divergence: `KL(π_true || π_pred)` 평균

**결과 저장 위치**:
```
results/
├── exp1_baseline/   eval_exp1_f81.txt, eval_exp1_f81_supervised.txt, eval_exp1_naive.txt
├── exp2_baseline_r/ eval_exp2_*.txt
├── exp3_gc/         eval_exp3_*.txt
├── exp4_gc_r/       eval_exp4_*.txt
├── exp5_gc/         eval_exp5_*.txt
└── exp6_gc_r/       eval_exp6_*.txt
```

완료 후 한꺼번에 확인:
```bash
cat results/exp{1,2,3,4,5,6}_*/eval_*.txt
```

---

## 디렉토리 구조

```
research_phyloGPN/
│
├── src/
│   ├── models/
│   │   ├── configuration.py               # PhyloGPNConfig
│   │   ├── model.py                       # PhyloGPNModel (RCEByteNet, ~83M params)
│   │   └── tokenizer.py                   # PhyloGPNTokenizer (A/C/G/T/N/-)
│   ├── losses/
│   │   ├── f81_loss.py                    # F81LikelihoodLoss
│   │   ├── f81_supervised_loss.py         # F81SupervisedLoss
│   │   └── supervised_loss.py             # SupervisedPiLoss (KL)
│   ├── data/
│   │   ├── dataset.py                     # SimF81Dataset (exp1, exp2)
│   │   ├── windowed_dataset.py            # WindowedSimF81Dataset (exp3-6)
│   │   └── collate.py
│   └── utils/
│       ├── math_f81.py                    # Felsenstein pruning (vectorized, rescaling)
│       ├── tree_utils.py                  # Newick 로더
│       └── checkpoint.py
│
├── data/
│   ├── simulate/
│   │   ├── simulate_exp1_baseline.py      # [Exp1] chunk π ~ Dirichlet, no rate (pyvolve)
│   │   ├── simulate_exp2_baseline_r.py    # [Exp2] chunk π ~ Dirichlet + per-site r (pyvolve)
│   │   ├── simulate_exp3_gc.py            # [Exp3/5] GC OU, no rate (custom F81)
│   │   ├── simulate_exp4_gc_r.py          # [Exp4/6] GC OU + per-site r (custom F81)
│   │   ├── fasta_to_npz.py               # FASTA + pi.txt → .npz
│   │   └── split_data.py                 # train/valid/test 분할 (exp1, exp2용)
│   ├── trees/
│   │   └── 241-mammalian-2020v2.1.nh.txt  # Zoonomia 247종 계통수
│   ├── exp1_baseline/{raw,processed}/
│   ├── exp2_baseline_r/{raw,processed}/
│   ├── exp3_gc/{raw,processed}/
│   ├── exp4_gc_r/{raw,processed}/
│   ├── exp5_gc/{raw,processed}/
│   └── exp6_gc_r/{raw,processed}/
│
├── scripts/
│   ├── save_exp1_split.py                 # exp1 split.json 사후 생성 (seed=42)
│   ├── exp1_baseline/{simulate,train_f81,train_f81_supervised,train_naive,eval}.sbatch
│   ├── exp2_baseline_r/{simulate,train_f81,train_f81_supervised,train_naive,eval}.sbatch
│   ├── exp3_gc/{simulate,train_f81_gc,train_f81_supervised_gc,train_naive_gc,eval}.sbatch
│   ├── exp4_gc_r/{simulate,train_f81_gc,train_f81_supervised_gc,train_naive_gc,eval}.sbatch
│   ├── exp5_gc/{simulate,train_f81_gc,train_f81_supervised_gc,train_naive_gc,eval}.sbatch
│   └── exp6_gc_r/{simulate,train_f81_gc,train_f81_supervised_gc,train_naive_gc,eval}.sbatch
│
├── checkpoints/
│   ├── exp1_baseline/{f81,f81_supervised,naive}/
│   ├── exp2_baseline_r/{f81,f81_supervised,naive}/
│   ├── exp3_gc/{f81,f81_supervised,naive}/
│   ├── exp4_gc_r/{f81,f81_supervised,naive}/
│   ├── exp5_gc/{f81,f81_supervised,naive}/
│   └── exp6_gc_r/{f81,f81_supervised,naive}/
│
├── results/
│   └── {exp_name}/eval_{exp}_{model}.txt
│
├── logs/
│   └── {exp_name}/
│
├── train_f81.py                           # [Exp1, Exp2] F81 훈련
├── train_f81_supervised.py                # [Exp1, Exp2] F81 Supervised 훈련
├── train_naive.py                         # [Exp1, Exp2] Naive 훈련
├── train_f81_gc.py                        # [Exp3-6] F81 훈련 (sliding window)
├── train_f81_supervised_gc.py             # [Exp3-6] F81 Supervised 훈련
├── train_naive_gc.py                      # [Exp3-6] Naive 훈련
├── evaluate.py                            # [Exp1, Exp2] 평가
└── evaluate_gc.py                         # [Exp3-6] 평가
```

---

## HPCC 환경

```
위치: /mnt/research/liulab/RA_lee/projects/research_phyloGPN
conda env: phylogpn-env
module: module purge && module load Miniforge3
```

## 워크플로우 (GitHub = 중심점)

```
로컬 코드 수정 → git push → HPCC git pull → HPCC에서 실행
HPCC 결과     → git add/commit/push → 로컬 git pull
```
