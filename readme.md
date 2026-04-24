# Project Setup Guide

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

## 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## 4. Download the MPE Environment

Download the MPE environment here: [Multi-Agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs).
Once downloaded, rename the folder to

```bash
mpe
```

Make sure to place it in the root directory of this project. This is required so that all import paths work correctly.

## 5. Delete the following folders

- GA_results_0.05
- GA_results_with_less_gens
- results_with_less_eps

## 6. Delete the CONTENT of the following folders. Make sure you <u>do not</u> delete the folder itself.

- GA_results
- results

## 6. Run the Project

After setup, you can run:

```bash
python main.py simple_spread  # MADDPG training
python evaluate.py simple_spread  # MADDPG evaluation

python GA_main.py simple_spread  # GA training
python GA_evaluate.py simple_spread  # GA evaluation
```

---
