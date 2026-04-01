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

## 네 가지 실험

| 실험 | π 설계 | rate | 데이터셋 클래스 | shortcut |
|------|--------|------|----------------|----------|
| **exp1_baseline** | chunk당 하나, Dirichlet(1,1,1,1) | 없음 | `SimF81Dataset` | ✓ (있음) |
| **exp2_baseline_r** | chunk당 하나, Dirichlet(1,1,1,1) | per-site r ~ Gamma(0.5) | `SimF81Dataset` | 부분적 |
| **exp3_gc** | GC OU process, position-varying | 없음 | `WindowedSimF81Dataset` | ✗ |
| **exp4_gc_r** | GC OU process, position-varying | per-site r ~ Gamma(0.5) | `WindowedSimF81Dataset` | ✗ |

### GC OU process 파라미터 (exp3, exp4)
```
GC(x) ~ Ornstein-Uhlenbeck(μ=0.41, σ=0.07, θ=1/100000)
GC range clipping: [0.30, 0.65]  (Bernardi isochore family 커버)
π(x) = (GC(x)/2, GC(x)/2, (1-GC(x))/2, (1-GC(x))/2)  # G,C,A,T
```

---

## 디렉토리 구조

```
research_phyloGPN/
│
├── src/                                   # 공유 패키지 (모든 실험 공통)
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
│   │   ├── windowed_dataset.py            # WindowedSimF81Dataset (exp3, exp4)
│   │   └── collate.py
│   └── utils/
│       ├── math_f81.py                    # Felsenstein pruning (vectorized)
│       ├── tree_utils.py                  # Newick 로더
│       └── checkpoint.py
│
├── data/
│   ├── simulate/
│   │   ├── simulate_exp1_baseline.py      # [Exp1] chunk π ~ Dirichlet, no rate
│   │   ├── simulate_exp2_baseline_r.py    # [Exp2] chunk π ~ Dirichlet + per-site r
│   │   ├── simulate_exp3_gc.py            # [Exp3] GC OU, no rate
│   │   ├── simulate_exp4_gc_r.py          # [Exp4] GC OU + per-site r
│   │   ├── fasta_to_npz.py               # FASTA + pi.txt → .npz
│   │   └── split_data.py                 # train/valid/test 분할 (exp1, exp2용)
│   ├── trees/
│   │   └── 241-mammalian-2020v2.1.nh.txt
│   ├── exp1_baseline/{raw,processed}/
│   ├── exp2_baseline_r/{raw,processed}/
│   ├── exp3_gc/{raw,processed}/
│   └── exp4_gc_r/{raw,processed}/
│
├── scripts/
│   ├── exp1_baseline/{simulate,train_f81,train_f81_supervised,train_naive}.sbatch
│   ├── exp2_baseline_r/{simulate,train_f81,train_f81_supervised,train_naive}.sbatch
│   ├── exp3_gc/{simulate,train_f81_gc,train_f81_supervised_gc,train_naive_gc}.sbatch
│   └── exp4_gc_r/{simulate,train_f81_gc,train_f81_supervised_gc,train_naive_gc}.sbatch
│
├── checkpoints/
│   ├── exp1_baseline/{f81,f81_supervised,naive}/
│   ├── exp2_baseline_r/{f81,f81_supervised,naive}/
│   ├── exp3_gc/{f81,f81_supervised,naive}/
│   └── exp4_gc_r/{f81,f81_supervised,naive}/
│
├── logs/
│   ├── exp1_baseline/
│   ├── exp2_baseline_r/
│   ├── exp3_gc/
│   └── exp4_gc_r/
│
├── train_f81.py                           # [Exp1, Exp2] F81 훈련
├── train_f81_supervised.py                # [Exp1, Exp2] F81 Supervised 훈련
├── train_naive.py                         # [Exp1, Exp2] Naive 훈련
├── train_f81_gc.py                        # [Exp3, Exp4] F81 훈련 (sliding window)
├── train_f81_supervised_gc.py             # [Exp3, Exp4] F81 Supervised 훈련
├── train_naive_gc.py                      # [Exp3, Exp4] Naive 훈련
└── evaluate.py                            # 평가: π_pred vs π_true
```

---

## 파이프라인 (실험별)

### Exp1 / Exp2 — chunk-level π (SimF81Dataset)

```
[Step 1] 시뮬레이션
  sbatch --array=0-999 scripts/exp{1,2}_*/simulate.sbatch
  → data/exp{1,2}_*/raw/chunk_NNNNN.{fasta,_pi.txt}
  → data/exp{1,2}_*/processed/block_NNNNN.npz

[Step 2] 훈련 (세 모델 독립 제출)
  sbatch scripts/exp1_baseline/train_f81.sbatch
  sbatch scripts/exp1_baseline/train_f81_supervised.sbatch
  sbatch scripts/exp1_baseline/train_naive.sbatch
```

npz 파일 구조: `ref_seq (481,)`, `pi_true (481,4)` (동일 π 반복), `msa_codes (481,S)`, `taxon_names (S,)`

### Exp3 / Exp4 — GC continuous π (WindowedSimF81Dataset)

```
[Step 1] 시뮬레이션
  sbatch --array=0-999 scripts/exp{3,4}_*/simulate.sbatch
  → data/exp{3,4}_*/raw/genome_NNNNN.{fasta,_pi.txt}
  → data/exp{3,4}_*/processed/genome_NNNNN.npz

[Step 2] 훈련 (세 모델 독립 제출)
  sbatch scripts/exp3_gc/train_f81_gc.sbatch
  sbatch scripts/exp3_gc/train_f81_supervised_gc.sbatch
  sbatch scripts/exp3_gc/train_naive_gc.sbatch
```

npz 파일 구조: `ref_seq (L,)`, `pi_true (L,4)` (position마다 다른 π), `msa_codes (L,S)`, `taxon_names (S,)`  
→ WindowedSimF81Dataset이 stride=1 sliding window로 분해

---

## HPCC 마이그레이션 노트 (Exp1)

현재 Exp1 훈련 job은 구버전 경로로 실행 중:
- 데이터: `data/processed/` → 완료 후 `data/exp1_baseline/processed/`로 이동
- 체크포인트: `checkpoints/f81/` → `checkpoints/exp1_baseline/f81/`로 이동
- 로그: `logs/train_f81_*.out` → `logs/exp1_baseline/`로 이동

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

**RCE**: DNA 양쪽 가닥 동등성을 weight parametrization으로 강제  
**Input**: ref_seq (L + 480 padding) → **Output**: π(4) per position (valid conv)

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
