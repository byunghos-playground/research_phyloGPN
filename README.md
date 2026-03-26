# research_phyloGPN

**PhyloGPN F81 Framework 검증 — Simulation Study**

Albors et al. 2025 (RECOMB) *A Phylogenetic Approach to Genomic Language Modeling* 에서 제안한 F81 phylogenetic loss framework를 **시뮬레이션 데이터**로 검증하는 프로젝트.

> 논문 저자들이 F81 loss 구현 코드를 공개하지 않아 수식 기반으로 직접 구현.

---

## 연구 목표

| 모델 | 설명 | 필요 데이터 |
|------|------|-------------|
| **모델 A (F81)** | F81 phylogenetic likelihood loss로 훈련 | ref_seq + MSA + 계통수 |
| **모델 B (Naive)** | π_true를 label로 직접 지도 학습 (baseline) | ref_seq + π_true |

두 모델 모두 시뮬레이션된 데이터에서 **각 사이트의 F81 stationary frequency π = (π_A, π_C, π_G, π_T)** 를 예측.

**핵심 질문**: Alignment + 계통수를 통한 간접 학습(F81)이 직접 지도 학습(Naive)보다 나은가?

---

## 디렉토리 구조

```
research_phyloGPN/
│
├── src/                          # 핵심 Python 패키지
│   ├── models/
│   │   ├── configuration.py      # PhyloGPNConfig (하이퍼파라미터)
│   │   ├── model.py              # PhyloGPNModel (RCEByteNet 아키텍처)
│   │   └── tokenizer.py          # PhyloGPNTokenizer (A/C/G/T/N/- 6종)
│   │
│   ├── losses/
│   │   ├── f81_loss.py           # F81LikelihoodLoss (Felsenstein pruning 기반)
│   │   └── supervised_loss.py    # SupervisedPiLoss (KL divergence)
│   │
│   ├── data/
│   │   ├── dataset.py            # SimF81Dataset (블록 단위 .npz 로더)
│   │   ├── windowed_dataset.py   # WindowedSimF81Dataset (sliding window)
│   │   └── collate.py            # DataLoader collate 함수
│   │
│   └── utils/
│       ├── math_f81.py           # F81 수학 (vectorized Felsenstein pruning)
│       ├── tree_utils.py         # TreeStruct, Newick 로더
│       └── checkpoint.py         # 체크포인트 저장/로드
│
├── data/
│   ├── simulate/
│   │   ├── simulate_f81.py       # [Step 1] pyvolve F81 시뮬레이션
│   │   ├── fasta_to_npz.py       # [Step 2] FASTA+pi → .npz 변환
│   │   └── split_data.py         # [Step 3] train/valid/test 분할
│   │
│   ├── trees/
│   │   └── 241-mammalian-2020v2.1.nh.txt   # Zoonomia 241종 계통수
│   │
│   ├── raw/        # simulate_f81.py 출력 (chunk_NNN.fasta, chunk_NNN_pi.txt)
│   ├── processed/  # fasta_to_npz.py 출력 (block_NNN.npz)
│   ├── train/      # split_data.py 출력
│   ├── valid/
│   └── test/
│
├── scripts/
│   ├── simulate.sbatch           # SLURM array job (시뮬레이션)
│   ├── train_f81.sbatch          # SLURM (모델 A 훈련)
│   └── train_naive.sbatch        # SLURM (모델 B 훈련)
│
├── train_f81.py                  # 훈련 엔트리포인트 — 모델 A
├── train_naive.py                # 훈련 엔트리포인트 — 모델 B
└── evaluate.py                   # 평가: π_pred vs π_true
```

---

## 데이터 파이프라인

### Step 0: 계통수 준비
```
data/trees/241-mammalian-2020v2.1.nh.txt
```
Zoonomia Consortium 241 포유류 계통수. 이전 작업(`2_loss_F81/F81_vanilla/data/`)에서 복사.

### Step 1: F81 시뮬레이션
각 사이트마다 독립적인 π를 Dirichlet(1,1,1,1)에서 샘플하여 pyvolve로 시뮬레이션.

```bash
# 단일 실행 (테스트용)
python data/simulate/simulate_f81.py \
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \
    --L 10000 \
    --out_prefix data/raw/chunk_000 \
    --seed 0

# SLURM array job (10 chunks = 100,000 사이트)
sbatch --array=0-9 scripts/simulate.sbatch
```

출력: `data/raw/chunk_NNN.fasta`, `data/raw/chunk_NNN_pi.txt`

### Step 2: .npz 변환
```bash
python data/simulate/fasta_to_npz.py \
    --fasta data/raw/chunk_000.fasta \
    --pi    data/raw/chunk_000_pi.txt \
    --out   data/processed/block_000.npz
```

`.npz` 파일 내용:
| 키 | Shape | 설명 |
|----|-------|------|
| `ref_seq` | `str (L,)` | 레퍼런스 종 뉴클레오타이드 서열 |
| `pi_true` | `(L, 4)` | 실제 F81 stationary frequency |
| `msa_codes` | `(L, S)` | 전체 alignment 정수 코드 |
| `taxon_names` | `(S,)` | 종 이름 (계통수 leaf order와 동일) |

### Step 3: Train/Valid/Test 분할
```bash
python data/simulate/split_data.py \
    --processed_dir data/processed \
    --train_ratio 0.8 --valid_ratio 0.1 --seed 42
```

---

## 훈련

### 모델 A: F81 Loss
```bash
# 로컬 테스트
python train_f81.py \
    --train_dir data/train \
    --valid_dir data/valid \
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \
    --out_dir   checkpoints/f81

# HPCC SLURM
sbatch scripts/train_f81.sbatch
```

### 모델 B: Naive (supervised baseline)
```bash
python train_naive.py \
    --train_dir data/train \
    --valid_dir data/valid \
    --out_dir   checkpoints/naive

# HPCC SLURM
sbatch scripts/train_naive.sbatch
```

---

## 평가

```bash
# 모델 A 평가
python evaluate.py \
    --checkpoint checkpoints/f81/best.pt \
    --test_dir   data/test \
    --model_name f81

# 모델 B 평가
python evaluate.py \
    --checkpoint checkpoints/naive/best.pt \
    --test_dir   data/test \
    --model_name naive
```

결과: `results/eval_f81.json`, `results/eval_naive.json`

평가 지표:
- **MAE**: |π_pred - π_true| 각 염기별 평균
- **Pearson r**: π_pred vs π_true 상관계수
- **KL divergence**: KL(π_true || π_pred)

---

## 구현 핵심 (기술 메모)

### F81 Transition Probability
```
P_ij(t) = π_j + exp(-μt) * (δ_ij - π_j)
```

### Felsenstein Pruning 핵심 공식 (vectorized)
```
contrib_k = (1 - e^{-μt}) * (π · L_child) + e^{-μt} * L_child[k]
```
→ 행렬 곱 없이 dot product + scalar 연산만으로 O(4) 계산 가능.
모든 (B, L) 사이트를 PyTorch 텐서 연산으로 동시 처리.

### 버그 수정 사항 (이전 버전 대비)
| 위치 | 버그 | 수정 |
|------|------|------|
| `math_f81.py` | postorder loop에서 root skip → `L_node[root]=None` → crash | root도 동일하게 처리 |
| `math_f81.py` | Python B×L 이중 for loop → 매우 느림 | PyTorch 벡터화 |
| `tokenizer.py` | `_convert_id_to_token`: `_vocab[int]` → KeyError | `_ID_TO_CHAR` 역방향 dict 추가 |

---

## HPCC 환경

```
위치: /mnt/research/liulab/RA_lee/projects/research_phyloGPN
conda env: phylogpn-env
module: module purge && module load Miniforge3
```

### conda 환경 생성
```bash
conda create -n phylogpn-env python=3.10
conda activate phylogpn-env
pip install torch torchvision torchaudio
pip install transformers numpy scipy pyvolve ete3
```

---

## 워크플로우 (GitHub = 중심점)

```
로컬 코드 수정 → git push → HPCC git pull → HPCC에서 실행
HPCC 결과 → git add/commit/push → 로컬 git pull
```
