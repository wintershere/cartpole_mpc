# Cart-Pole NMPC with acados
## Exercise 10 — Diehl NOC Course (Winter 2025/26)

---

## Step 0: Prerequisites
Make sure you have Python 3.9+ and pip.

---

## Step 1: Install CasADi (fast, no compilation)
```bash
pip install casadi numpy matplotlib scipy
```

---

## Step 2: Install acados (requires compilation — do this first!)

### Option A: From source (recommended, ~5 min)
```bash
# Go to wherever you want to install acados (NOT inside this folder)
cd ~
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init

# Build
mkdir build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON
make install -j4

# Set environment variable (add this to your ~/.bashrc too!)
export ACADOS_SOURCE_DIR=~/acados
```

### Install the Python interface
```bash
cd ~/acados/interfaces/acados_template
pip install -e .
```

### Test it works
```bash
cd ~/acados
python examples/acados_python/getting_started/minimal_example_ocp.py
# Should run without errors and print timing results
```

---

## Step 3: Run the assignment

```bash
cd ~/cartpole_mpc   # wherever you saved these files

# Test model builds correctly
python cartpole_model.py

# Test OCP solver compiles and solves once
python cartpole_ocp.py

# Run the full MPC closed-loop (Exercise 3a — no noise, no RTI)
python cartpole_closed_loop.py
```

---

## Step 4: Run experiments

Edit the CONFIG section at the top of `cartpole_closed_loop.py`:

### Exercise 3a — Basic MPC
```python
USE_RTI     = False
ADD_NOISE   = False
ADD_IMPULSE = False
```

### Exercise 3b — With disturbances
```python
USE_RTI     = False
ADD_NOISE   = True
ADD_IMPULSE = True
```

### Exercise 4 — RTI scheme
```python
USE_RTI     = True
ADD_NOISE   = False
ADD_IMPULSE = False
```

---

## What to look for in results

| Experiment | What to observe |
|------------|-----------------|
| 3a (clean) | θ goes from π → 0 (swing-up). Note mean solve time vs dt=40ms |
| 3b (noise) | Does it still swing up? How does impulse affect it? |
| 4 (RTI)    | Solve time should drop ~10x. Does trajectory quality change? |

---

## File overview

| File | Purpose |
|------|---------|
| `cartpole_model.py` | Defines cart-pole dynamics as AcadosModel |
| `cartpole_ocp.py`   | Sets up and compiles the OCP solver |
| `cartpole_closed_loop.py` | Runs the full MPC loop, plots results |

---

## Common errors

**`ACADOS_SOURCE_DIR not set`**
→ Run: `export ACADOS_SOURCE_DIR=~/acados`

**`status 2` warning during simulation**
→ Solver didn't converge in max iterations. The returned u0 is from a non-converged solution.
→ Try: increase `Fmax`, adjust Q/R, or reduce noise magnitude.

**Compilation takes forever**
→ Normal the first time (~15-30s). Subsequent runs use cached C code.

**Plot doesn't show**
→ Check `cartpole_results.png` in the same folder.
