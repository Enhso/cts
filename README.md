# Context Tree Switching

A Python implementation of Context Tree Switching (CTS) and, as a baseline and
cross-check, Context Tree Weighting (CTW), following Veness, Ng, Hutter and
Bowling, *Context Tree Switching*, DCC 2012 (arXiv:1111.3182). On top of the two
models there is an integer arithmetic coder, a file compressor, and a benchmark
on the Calgary corpus. Standard library only; pytest is needed just for the tests.

## What is implemented

- **KT estimator** (`src/cts_core/kt_estimator.py`): Eq. 7, with a sequential
  `KTEstimator` that is the depth-0 case of both trees.
- **Context tree** (`src/cts_core/context_tree.py`): nodes created on demand. A
  context is the preceding bits, most recent first, so the newest bit picks the
  first child below the root.
- **CTW and CTS** (`src/cts_core/cts_model.py`): sequential predictors.
  `predict()` gives P(next bit = 1 | history), `update(bit)` observes a bit,
  `log_joint` is ln of the probability of everything seen, and `code_length` is
  that probability's -log2 in bits. CTW follows Eq. 14. CTS follows the node
  update equations of section 3.1 with the paper's switching rate
  alpha_{n+1} = 1/(n+1). Every statistic is kept as a logarithm, so long
  sequences do not underflow.
- **Arithmetic coder** (`src/cts_core/arithmetic_coder.py`): a 32-bit binary
  coder (Witten, Neal and Cleary) with 16-bit probabilities.
- **Codec and CLI** (`src/cts_core/codec.py`, `src/main.py`): bytes are split
  into bits, most significant first, and each bit is coded under the model. A
  14-byte header stores the model, depth and length.

```python
from src.cts_core.cts_model import CTSModel

model = CTSModel(depth=8)
for bit in [0, 1, 1, 0, 1, 1, 0, 1]:
    p1 = model.predict()      # P(next bit = 1 | history)
    model.update(bit)
print(model.code_length)      # -log2 of the joint probability, in bits
```

## Install, test, run

Tested on Python 3.12.

```
python3 -m venv venv_cts && source venv_cts/bin/activate
pip install -r requirements.txt            # pytest only
python -m pytest -q                        # 151 tests, about 20 s
```

```
python -m src.main compress   input output [--model cts|ctw] [--depth 48]
python -m src.main decompress input output
python -m src.main bench      file [file ...] [--depth 48] [--limit BYTES]
```

`python src/main.py ...` works too. `--depth` is the context length in bits
(0 to 255, default 48 as in the paper). `bench` prints average bits per byte of
CTW and CTS from the ideal code length, -log2 P(data); `--limit` uses only the
first BYTES bytes of each file, which is how to time a slice before a long run.

## Tests

- KT: the sequential product matches its closed form (exact integers) for every
  binary sequence up to length 10, and the integral definition of Eq. 6.
- Depth-0 CTW and CTS reduce to KT, step by step.
- CTW and CTS match naive, non-incremental implementations of Eq. 14 and Eq. 16
  (`tests/naive_reference.py`, sharing no code with `src/`) on **every binary
  sequence of length up to 10 at depths 0 to 3**. The worst disagreement in ln P
  is about 6e-15.
- Those comparisons have teeth: with the model's switching rate scaled by 0.9,
  shifted by one time step, or halved, 220 of 254 sequences (length up to 7,
  depth 2) disagree, and the exhaustive test itself fails; the same goes for a
  CTW weight of 0.6 instead of 1/2, and for perturbing the reference instead of
  the model.
- P(0) + P(1) = 1 at every step; `log_prob` is read-only and agrees with `update`;
  code length equals -log2 of the joint probability.
- A constant sequence costs logarithmically: for n up to 8192 the code length
  stays under the paper's own bounds (Eq. 15 for CTW, Eq. 17 for CTS).
- CTS with the switching rate set to 0 equals CTW; 100,000 random bits do not
  underflow; on 400-bit inputs the log-space models agree with 100-digit
  decimal arithmetic.
- compress then decompress restores random bytes, text, empty input, one-byte
  inputs, long runs and all 256 byte values, for both models at several depths,
  and the compressed size is within 3 bits plus byte padding of the ideal.

## Results

Depth 48 (six bytes of context), the setting of the paper's CTW48 and CTS48 rows
of Table 1, which are the base algorithms with no enhancements. Figures are
average bits per byte from the ideal code length, with the paper's Table 1
values next to them.

| file | bytes | CTW | CTW paper | CTS | CTS paper | CTW s | CTS s |
|---|---:|---:|---:|---:|---:|---:|---:|
| bib | 111261 | 2.245 | 2.25 | 2.241 | 2.23 | 43.4 | 54.2 |
| obj1 | 21504 | 4.657 | 4.63 | 4.612 | 4.70 | 8.3 | 11.4 |
| paper1 | 53161 | 2.807 | 2.84 | 2.806 | 2.78 | 19.7 | 33.9 |
| paper2 | 82199 | 2.560 | 2.59 | 2.559 | 2.56 | 38.7 | 52.3 |
| paper3 | 46526 | 2.931 | 2.97 | 2.931 | 2.95 | 19.2 | 27.0 |
| paper4 | 13286 | 3.448 | 3.50 | 3.449 | 3.48 | 4.9 | 6.9 |
| paper5 | 11954 | 3.704 | 3.73 | 3.705 | 3.70 | 4.2 | 5.7 |
| paper6 | 38105 | 2.964 | 2.99 | 2.963 | 2.93 | 13.8 | 20.2 |
| progc | 39611 | 2.961 | 3.00 | 2.958 | 2.94 | 21.9 | 24.9 |
| progl | 71646 | 2.102 | 2.11 | 2.097 | 2.05 | 30.2 | 39.3 |
| progp | 49379 | 2.213 | 2.24 | 2.203 | 2.12 | 16.0 | 23.3 |
| trans | 93695 | 2.074 | 2.09 | 2.072 | 1.95 | 34.0 | 43.8 |
| **weighted by size** | 632327 | 2.563 | 2.583 | 2.558 | 2.526 | | |

Runtimes are wall-clock seconds for the whole file, measured with two or three
runs sharing an 8-core machine. Peak memory was 0.76 to 1.14 GB per run (1.14 GB
for bib). The other six files (book1, book2, geo, news, obj2, pic) were not run,
so the paper's file-size-weighted averages (Table 2) are not comparable to the
weighted row above, which covers these twelve files only.

A real round trip: `compress` on paper5 with CTS at depth 48 gives 5551 bytes
(0.464 of the original; the ideal code length is 5536 bytes plus the 14-byte
header) in 12 s, and `decompress` restores a byte-identical file.

**What matches and what does not.** The absolute figures agree with the paper:
CTW is within 1.5% of the paper's CTW48 on all twelve files (lower on eleven),
and CTS is within 1.1% of the paper's CTS48 on eight of twelve. What I do not
reproduce is the margin of CTS over CTW. The paper has CTS ahead by 0.6% to
6.7% on eleven of these twelve files (behind on obj1); here CTS is between 0.03%
behind and 1.0% ahead, and the 1.0% is on obj1, where the paper's CTS is behind.
The big paper gains are not there: progp 5.4% in the paper against 0.45% here,
trans 6.7% against 0.10%, progl 2.8% against 0.24%, and progc, paper1 and paper6
about 2% against 0.03% to 0.10%. CTS comes out at 2.072 against the paper's 1.95
on trans, 2.203 against 2.12 on progp, and 2.097 against 2.05 on progl. The cause
is open. I ruled out a few things:

- The implementation. It equals the paper's Eq. 14 and Eq. 16 on every short
  sequence, and the 100-digit check agrees to about 1e-10 in ln P (values near
  -12000) at 32,000 bits, and to 1e-11 at depth 48 on 16,000 bits, so there is
  no numerical drift.
- The switching rate. The per-context rate 1/n_c that the paper says performs
  poorly is 11% worse here too (4.89 against 4.41 bits per byte on the first
  4000 bytes of paper5 at depth 48). Scaling the paper's rate up only makes
  things worse (progp, first 20000 bytes: CTW 2.828; CTS with the paper's rate
  2.826; with the rate times 2, 4, 8, 16, 64: 2.828, 2.833, 2.844, 2.868, 3.009).
- Bit order. Least significant bit first is slightly worse for both models
  (progp, first 20000 bytes: CTW 2.863, CTS 2.863, against 2.828 and 2.826).

I could not check the authors' code: the URL in the paper's footnote now returns
404. Unstated details in their implementation are the likeliest explanation, but
that is a guess.

## Deviations from the paper

Where the task brief and the paper differ, this implementation follows the paper;
I found no substantive difference between them. These are the points where the
paper needed interpreting or where this code does something different:

1. **Switching-rate index.** The paper says alpha^c_n = n^-1 "for any
   sub-context" and updates with alpha^c_{n+1}, so n is the position in the whole
   bit sequence, not the number of visits to the node. I use alpha = 1/(n+1)
   after the n-th bit at every node. Consistently, in the recursion of Eq. 16 the
   step between the (k-1)-th and k-th symbol of a context uses the rate of the
   time the (k-1)-th one occurred, plus one. The step in the proof of Theorem 3
   that bounds the weight from below needs that rate to be at most 1/k, which
   holds.
2. **First bits.** The paper holds back the first D bits and codes them
   separately. Here the context is padded with zeros, as if the data had been
   preceded by D zero bits, so every bit is coded and the tree always has full
   depth. The difference is at most D bits per file.
3. **Update path.** The list of nodes to update in section 3.1 is written with
   phi_D(x_{1:n}); I use phi_D(x_{<n}), the context used to predict x_n, as the
   traversal rule in the same section says.
4. **No enhancements.** Only the base CTW48/CTS48 are implemented. The paper's
   starred variants (binary decomposition after Willems and Tjalkens, count
   scaling by 0.98, k and s initialised to 0.925 and 0.075, the zero-redundancy
   estimator, CTS at depth 160) are not, so the starred rows of Table 1 are not
   comparable.
5. **Bit order** within a byte is not stated in the paper; most significant bit
   first is used.
6. **Reported figures** are the ideal code length -log2 P(data), without the
   header or arithmetic-coding overhead (under 3 bits plus byte padding), not the
   size of compressed files as in the paper.
7. **Probabilities** are rounded to 16 bits for the coder, so a bit can never cost
   less than about 2e-5 bits.
8. The model's `log_prob` applies the update and rolls it back rather than
   computing the prediction in a separate pass, so it cannot disagree with
   `update`.

## Known limits

- **Speed.** Pure Python at depth 48: 0.48 ms per byte for CTS and 0.33 ms for
  CTW measured alone on an 8000-byte slice, and 0.5 to 0.65 and 0.4 to 0.5 ms per
  byte in the full runs above. Roughly 2 KB/s. `compress` and `decompress` take about
  twice the model time (24.7 s for the round trip on paper5) because predicting
  a bit evaluates the update once and applying it evaluates it again. Extrapolating
  from those rates, the whole 3.1 MB corpus would take 45 to 60 minutes of CPU at
  depth 48 in `bench`.
- **Memory.** The tree is never pruned and grows by up to depth + 1 nodes per
  input bit. At depth 48 the peak was about 10 KB of RAM per input byte or more
  on these files (1.14 GB for the 111 KB bib), so book1 (768 KB) would need around
  8 GB or more. That, more than time, is why the large files were skipped.
- Compressed files are only guaranteed to decode with the same Python and
  floating-point library that wrote them: the coder trusts the model's
  `exp`/`log` results to match between runs.
- No checksum in the file format. Depth is limited to 255, and files are held
  fully in memory.

## Getting the corpus

`data/calgary_corpus/` is git-ignored and was filled from the Canterbury Corpus
site:

```
mkdir -p /tmp/calgary && cd /tmp/calgary
curl -O https://corpus.canterbury.ac.nz/resources/calgary.tar.gz
tar -tzf calgary.tar.gz                     # 18 plain file names, no paths
tar -xzf calgary.tar.gz --no-same-owner --no-same-permissions
cp * /path/to/cts/data/calgary_corpus/
```

The archive's SHA-256 was
`e109eebdc19c5cee533c58bd6a49a4be3a77cc52f84ba234a089148a4f2093b7`.
Then, for example:

```
python -m src.main bench data/calgary_corpus/paper5 --depth 48
```
