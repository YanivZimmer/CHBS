
# **Official implementation of “Supervised Embedded Methods for Hyperspectral Band Selection”**  
*Yaniv Zimmer, Ofir Lindenbaum, and Oren Glickman — ECAI (2025)*

[Read the paper](https://ebooks.iospress.nl/doi/10.3233/FAIA251180)

[ArXiv](https://arxiv.org/abs/2401.11420)


## CHBS: Concrete Hyperspectral Band Selection

---

## 🌈 Overview

**CHBS (Concrete Hyperspectral Band Selection)** is a supervised, embedded method for **task-specific hyperspectral band selection**.

Unlike traditional preprocessing-based methods, CHBS integrates band selection **directly into a deep learning model** through a **Concrete Selector Layer** using **Gumbel–Softmax reparameterization**.

This design enables differentiable, end-to-end training that jointly optimizes both **band selection and downstream task performance**, making CHBS ideal for **real-time or resource-constrained environments** such as autonomous driving and environmental sensing.

---

## 🔍 Key Features

- 🚀 **Embedded Band Selection:** No separate preprocessing — fully end-to-end.  
- 🧩 **Concrete Selector Layer:** Differentiable band selection via Gumbel–Softmax.  
- 🎯 **Segmented Xavier Initialization:** Ensures spectral diversity and stable convergence.  
- 🧠 **Task-Aware Optimization:** Selects bands that improve model accuracy directly.  
- 📈 **State-of-the-Art Results:** Outperforms prior methods on multiple HSI benchmarks.

---

## 🧠 Method Summary

CHBS learns *k* informative spectral bands by training a learnable logits matrix **L** via a Gumbel–Softmax distribution:

$$
M_{i,j} = \frac{\exp((L_{i,j} + G_{i,j}) / \tau)}{\sum_r \exp((L_{i,r} + G_{i,r}) / \tau)}
$$


This produces a soft, differentiable band-selection mask that becomes nearly discrete as the temperature τ decreases during training.

---

## ⚙️ Installation
git clone https://github.com/<your-username>/CHBS.git
cd CHBS
pip install -r requirements.txt


## 🧩 Acknowledgements

This work was partially supported by the Chief Scientist of the Israeli Ministry of Agriculture, grant number 12-03-0010.

## 📚 Citation

please cite:

```bibtex
@article{zimmer2024embedded,
  title={Embedded hyperspectral band selection with adaptive optimization for image semantic segmentation},
  author={Zimmer, Yaniv and Glickman, Oren},
  journal={arXiv preprint arXiv:2401.11420},
  year={2024}
}
