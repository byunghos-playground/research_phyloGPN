# research_phyloGPN

**PhyloGPN F81 Framework 검증 — Simulation Study**

Albors et al. 2025 (RECOMB) *A Phylogenetic Approach to Genomic Language Modeling* 에서 제안한
F81 phylogenetic loss framework를 **시뮬레이션 데이터**로 검증하는 프로젝트.

> 논문 저자들이 F81 loss 구현 코드를 공개하지 않아 논문 수식 기반으로 직접 구현.

---

## PhyloGPN이란

### 모델이 하는 일

각 genomic site마다 **F81 stationary frequency π = (π_A, π_C, π_G, π_T)** 를 예측.
π는 "이 사이트에서 evolution을 지배하는 염기 분포" — 어떤 염기가 진화적으로 선호되는지를 나타냄.

### 입력과 출력

- **Input**: reference genome의 연속된 481bp (center site 기준 좌우 240bp 포함)
- **Output**: center site 하나의 π = (π_A, π_C, π_G, π_T)

전체 서열은 center를 1bp씩 이동하는 **sliding window** 방식으로 처리.
center c마다 ref_seq[c-240:c+241] (481bp) 를 모델에 넣어 π_pred 1개를 얻음.

### 왜 481bp 맥락이 필요한가

center 1개 nucleotide만 보면 그 위치가 coding region인지, CpG site인지, conserved region인지 알 수 없음.
주변 480bp를 같이 보면 맥락에서 evolutionary constraint를 추론할 수 있음.

모델 아키텍처(RCEByteNet, 40개 dilated conv block)가 한 번에 볼 수 있는 범위가 정확히 481bp로 설계됨.
valid convolution을 사용하므로 481bp input → 1개 output (center site의 θ).

### F81 Loss — 어떻게 π를 label 없이 학습하는가

π_true 없이 **241종 alignment + 계통수**만으로 학습 신호를 만듦.

```
데이터:
  ref_seq[c-240:c+241]  → (481bp) →  모델  →  π_pred

  msa_codes[c, :]  = center site의 241종 nucleotide 관찰값
  tree             = 241-mammalian 계통수 (branch length 포함)
```

**Felsenstein Pruning**: tree의 각 leaf에 관찰값을 붙이고 bottom-up으로 올라가면서
"이 π_pred로 241종 관찰값이 나올 확률 P" 를 계산.

```
Human leaf  → [1,0,0,0]  (A 관찰)
Mouse leaf  → [1,0,0,0]  (A 관찰)
Dog   leaf  → [0,0,1,0]  (G 관찰)
...241개 leaf

leaf → internal node → ... → root
각 edge에서: contrib_k = (1-e^{-μt})·(π·L_child) + e^{-μt}·L_child[k]

root에서: P = Σ_k π_k × L_root[k]  →  숫자 1개
```

Tree가 없으면 241개 관찰값을 독립으로 취급하게 됨. Tree의 branch length가
"Human-Mouse는 가까우니까 비슷해도 정보가 적고, Human-Fish는 멀어서 같으면 conserved 신호"
를 반영.

**Loss** (논문 Eq.4):
```
L = -log P_F81(alignment | π_pred, T)   ← P가 높을수록 loss 낮음
  + log π_ref                           ← conditioning term
```

conditioning term은 shortcut 방지: ref_seq가 모델 input이면서 MSA leaf 0에도 있으므로,
모델이 "ref에서 A 관찰 → π_A를 높게 예측" shortcut을 쓸 수 있음.
`+log π_ref`를 loss에 더하면 π_ref가 높을수록 loss가 커져서 이 shortcut을 penalize.
결과적으로 모델은 다른 240종 alignment에서 신호를 받아야 함.

### PhyloGPN vs 우리 Simulation Study

| | PhyloGPN 논문 | 우리 Simulation Study |
|--|--|--|
| 데이터 | 실제 Zoonomia 241종 게놈 | pyvolve 시뮬레이션 |
| π_true | 알 수 없음 | **알 수 있음** (시뮬레이션 파라미터) |
| 목적 | genomic representation 학습 | F81 loss가 π를 제대로 복원하는지 검증 |

시뮬레이션에서 π_true를 알고 있으므로 훈련 후 π_pred vs π_true 직접 비교 가능.

---

## 연구 목표 — 세 가지 실험

| 모델 | Loss | π_true 사용 | Alignment 사용 | 분류 |
|------|------|-------------|----------------|------|
| **F81** | `-log P_F81(alignment \| π_pred, T) + log π_ref` | ✗ | ✓ | PIML (unsupervised) |
| **F81 Supervised** | `log P_F81(alignment \| π_true, T) - log P_F81(alignment \| π_pred, T)` | ✓ (간접) | ✓ | PIML (supervised) |
| **Naive** | `KL(π_true \|\| π_pred)` | ✓ (직접) | ✗ | Supervised baseline |

**F81 Supervised**는 "π_true로 계산한 oracle likelihood를 target으로, π_pred의 likelihood가 그걸 따라가도록 학습".
π를 직접 비교하는 게 아니라 likelihood를 통해 간접적으로 π_true 정보를 활용하는 Physics-Informed 방식.

**핵심 질문**: alignment + tree의 evolutionary signal로 학습한 F81이 직접 지도 학습(Naive)에 비해 어떤가?

---

## 아키텍처: RCEByteNet

| 항목 | 값 |
|------|-----|
| 채널 수 (outer_dim) | 960 |
| Bottleneck (inner_dim) | 480 |
| Kernel size | 5 |
| Dilation 패턴 | [1, 5] × 20 = 40 blocks |
| Receptive field | 481 bp |
| 파라미터 수 | ~83M |

**RCE (Reverse-Complement Equivariance)**: DNA 양쪽 가닥이 동등해야 함.
Weight parametrization으로 강제: `W_rc = (W + flip(W_complement)) / 2`

**Vocab**: A=0, C=1, G=2, T=3, N=4(unknown), -=5(gap/pad)

---

## 디렉토리 구조

```
research_phyloGPN/
│
├── src/                               # 핵심 Python 패키지
│   ├── models/
│   │   ├── configuration.py           # PhyloGPNConfig (하이퍼파라미터)
│   │   ├── model.py                   # PhyloGPNModel (RCEByteNet 아키텍처)
│   │   └── tokenizer.py               # PhyloGPNTokenizer (A/C/G/T/N/- 6종)
│   │
│   ├── losses/
│   │   ├── f81_loss.py                # F81LikelihoodLoss — NLL + conditioning term
│   │   ├── f81_supervised_loss.py     # F81SupervisedLoss — likelihood matching
│   │   └── supervised_loss.py         # SupervisedPiLoss — KL(π_true || π_pred)
│   │
│   ├── data/
│   │   ├── dataset.py                 # SimF81Dataset (청크 단위 .npz 로더, 세 모델 공통)
│   │   ├── windowed_dataset.py        # WindowedSimF81Dataset (현재 미사용 — 실제 게놈 데이터용)
│   │   └── collate.py                 # DataLoader collate 함수 2종
│   │
│   └── utils/
│       ├── math_f81.py                # F81 수학 (vectorized Felsenstein pruning)
│       ├── tree_utils.py              # TreeStruct, Newick 로더 (ete3)
│       └── checkpoint.py              # 체크포인트 저장/로드/BestModelTracker
│
├── data/
│   ├── simulate/
│   │   ├── simulate_f81.py            # [Step 1] pyvolve F81 시뮬레이션
│   │   ├── fasta_to_npz.py            # [Step 2] FASTA+pi → .npz 변환
│   │   └── split_data.py              # [Step 3] train/valid/test 분할
│   │
│   ├── trees/
│   │   └── 241-mammalian-2020v2.1.nh.txt   # Zoonomia 241종 계통수
│   │
│   ├── raw/        # simulate_f81.py 출력 (chunk_NNN.fasta, chunk_NNN_pi.txt)
│   ├── processed/  # fasta_to_npz.py 출력 (block_NNN.npz)
│   ├── train/      # split_data.py 출력 (심볼릭 링크)
│   ├── valid/
│   └── test/
│
├── scripts/
│   ├── simulate.sbatch                # SLURM array job (시뮬레이션)
│   ├── train_f81.sbatch               # SLURM (F81 훈련)
│   ├── train_f81_supervised.sbatch    # SLURM (F81 Supervised 훈련)
│   └── train_naive.sbatch             # SLURM (Naive 훈련)
│
├── train_f81.py                       # 훈련 엔트리포인트 — F81
├── train_f81_supervised.py            # 훈련 엔트리포인트 — F81 Supervised
├── train_naive.py                     # 훈련 엔트리포인트 — Naive
└── evaluate.py                        # 평가: π_pred vs π_true (3개 모델 공통)
```

---

## 데이터 파이프라인

### 전체 흐름

```
[Step 1+2] simulate.sbatch (SLURM array, 병렬)
  simulate_f81.py  →  chunk_NNN.fasta + chunk_NNN_pi.txt
  fasta_to_npz.py  →  data/processed/block_NNN.npz

[Step 3] split_data.py (array 완료 후 1회 실행)
  data/processed/  →  data/train/ + data/valid/ + data/test/

[Step 4] 훈련 (세 모델 독립적으로 제출)
  train_f81.sbatch
  train_f81_supervised.sbatch
  train_naive.sbatch
```

**권장 규모**: 10,000 chunks × 481 sites (청크당 π 하나)
- 각 청크 = 독립적인 π 하나 → 다양한 π 커버리지 확보
- array job은 병렬 실행 → 청크 수와 무관하게 벽시계 ~1시간
- 분할: train 8,000 / valid 1,000 / test 1,000 청크

### 데이터 구조

각 `.npz` 블록 파일 내용:

| 키 | Shape | 설명 |
|----|-------|------|
| `ref_seq` | `str (L,)` | 첫 번째 종(ref)의 뉴클레오타이드 서열 — 모델 input |
| `pi_true` | `(L, 4)` | 시뮬레이션에 사용된 실제 F81 π — 검증 ground truth |
| `msa_codes` | `(L, S)` | 241종 전체 alignment 정수 코드 — F81 loss용 |
| `taxon_names` | `(S,)` | 종 이름 배열 (tree leaf_order와 동일 순서) |

`msa_codes[:, 0]` = ref 종 (ref_seq와 동일 종). F81 loss의 conditioning term에서 사용.

### Step 1+2: 시뮬레이션 + npz 변환

**청크당 π 하나**: 각 청크는 L=481 사이트 전체가 동일한 π ~ Dirichlet(1,1,1,1)에서 생성.
ref_seq의 empirical frequency가 π 정보를 담아 모델이 context → π 매핑을 학습 가능.
`simulate.sbatch`가 시뮬레이션과 npz 변환을 한 번에 처리.

```bash
# SLURM array job (10,000 chunks 권장)
sbatch --array=0-9999 scripts/simulate.sbatch

# 단일 실행 (테스트용)
python data/simulate/simulate_f81.py \
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \
    --L 481 \
    --out_prefix data/raw/chunk_000 \
    --seed 0
python data/simulate/fasta_to_npz.py \
    --fasta data/raw/chunk_000.fasta \
    --pi    data/raw/chunk_000_pi.txt \
    --out   data/processed/block_000.npz
```

출력: `data/processed/block_NNN.npz` (청크 수만큼)

### Step 3: Train/Valid/Test 분할

array job 완료 후 1회 실행.

```bash
python data/simulate/split_data.py \
    --processed_dir data/processed \
    --train_ratio 0.8 --valid_ratio 0.1 --seed 42 \
    --copy    # HPCC NFS 환경에서는 symlink 대신 복사 권장
```

---

## 훈련

### 모델별 데이터 사용 방식

| 모델 | Dataset | 모델 Input | Loss에 필요한 것 |
|------|---------|-----------|----------------|
| F81 | `SimF81Dataset` | ref_seq (L + pad 480) | msa_codes[:, :] + tree |
| F81 Supervised | `SimF81Dataset` | ref_seq (L + pad 480) | msa_codes[:, :] + tree + π_true[:] |
| Naive | `SimF81Dataset` | ref_seq (L + pad 480) | π_true[:] |

세 모델 모두 `SimF81Dataset` 사용. 청크 1개 (L=481) → L개 π 예측 → L개 위치 전체에 loss 계산.

> `src/data/windowed_dataset.py` (`WindowedSimF81Dataset`)는 현재 미사용.
> 향후 실제 게놈 데이터 (긴 서열 + sliding window) 시 재사용 가능.

### F81

```bash
python train_f81.py \
    --train_dir data/train \
    --valid_dir data/valid \
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \
    --out_dir   checkpoints/f81

sbatch scripts/train_f81.sbatch
```

### F81 Supervised

```bash
python train_f81_supervised.py \
    --train_dir data/train \
    --valid_dir data/valid \
    --tree_path data/trees/241-mammalian-2020v2.1.nh.txt \
    --out_dir   checkpoints/f81_supervised

sbatch scripts/train_f81_supervised.sbatch
```

### Naive

```bash
python train_naive.py \
    --train_dir data/train \
    --valid_dir data/valid \
    --out_dir   checkpoints/naive

sbatch scripts/train_naive.sbatch
```

---

## 평가

세 모델 모두 동일한 `evaluate.py` 사용. `SimF81Dataset`으로 test 데이터 로드,
모델이 π_pred 예측 → π_true와 비교.

```bash
python evaluate.py \
    --checkpoint checkpoints/f81/best.pt \
    --test_dir   data/test \
    --model_name f81

python evaluate.py \
    --checkpoint checkpoints/f81_supervised/best.pt \
    --test_dir   data/test \
    --model_name f81_supervised

python evaluate.py \
    --checkpoint checkpoints/naive/best.pt \
    --test_dir   data/test \
    --model_name naive
```

결과: `results/eval_{model_name}.json`

평가 지표:
- **MAE**: |π_pred - π_true| 각 염기별 + 전체 평균
- **Pearson r**: π_pred vs π_true 각 염기별 + 전체 상관계수
- **KL divergence**: KL(π_true || π_pred) 평균

---

## 구현 핵심 (기술 메모)

### F81 전이 확률

```
P_ij(t) = π_j + exp(-μt) * (δ_ij - π_j)
```

### Felsenstein Pruning — vectorized 핵심 공식

```
contrib_k = (1 - e^{-μt}) * (π · L_child) + e^{-μt} * L_child[k]
```

행렬 곱 없이 dot product + scalar 연산만으로 O(4) 계산.
모든 (B, L) 사이트를 PyTorch 텐서 연산으로 동시 처리: `L_node (B, L, n_nodes, 4)`.

### 버그 수정 사항 (구버전 대비)

| 위치 | 버그 | 수정 |
|------|------|------|
| `math_f81.py` | postorder loop에서 root skip → `L_node[root]=None` → crash | root도 동일하게 처리 |
| `math_f81.py` | Python B×L 이중 for loop → 매우 느림 | PyTorch 벡터화 |
| `tokenizer.py` | `_convert_id_to_token`: `_vocab[int]` → KeyError | `_ID_TO_CHAR` 역방향 dict 추가 |
| `model.py` | `self.embedding.requires_grad = False` → no-op (Module attribute) | 삭제 |
| `train_f81.py` | model output (B,1,4) vs msa_codes (B,481,S) shape mismatch → ValueError | center column만 슬라이싱해서 전달 |
| `f81_loss.py` | conditioning term 누락 — NLL만 사용 | `+log π_ref` 추가 (논문 Eq.4 완성) |

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
HPCC 결과     → git add/commit/push → 로컬 git pull
```
