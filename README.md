# HannesImitation: Grasping with the Hannes Prosthetic Hand via Imitation Learning

<img src="https://github.com/user-attachments/assets/129ccc56-b621-445a-94e4-49a057a1e9f3" width="500">

_HannesImitation_ is an imitation learning-based approach that trains a single grasping policy across diverse objects and environments. 
The learned policy is deployed on the Hannes prosthetic hand, enabling control of wrist orientation and hand closure. The main contributions are
- _HannesImitationPolicy_: The novel application of Diffusion Policy to control a prosthetic hand equipped with and anthropomorphic wrist.
- _HannesImitationDataset_: A novel dataset for imitation learning with the Hannes prosthetic hand for grasping on unstructured scenarios.

### 🛠️ Installation
From the terminal, clone this repository with the following command. This repository uses Diffusion Policy as a submodule stored in `hannes_imitation/external/diffusion_policy`. This command also fetches the source code of all the submodules.

```
git clone --recurse-submodules https://github.com/hsp-iit/HannesImitation.git
cd HannesImitation
```

If you did a normal clone:

```
git clone https://github.com/hsp-iit/HannesImitation.git
cd HannesImitation
```

Git does not clone the source code of the submodules in you local directory by default. If you also want to fetch the source code of the submodules, then do

```
git submodule update --init --recursive
```

The _HannesImitation_ project builds on top of Diffusion Policy. To set up the environment and dependencies, please follow the installation instructions provided in the Diffusion Policy repository: [https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).

### 📚 Reading
Preprint available on *arXiv*: [https://arxiv.org/abs/2508.00491](https://arxiv.org/abs/2508.00491).

### 📂 Dataset
The HannesImitationDataset is available on HuggingFace at: [https://huggingface.co/datasets/HSP-IIT/HannesImitation](https://huggingface.co/datasets/HSP-IIT/HannesImitation)

### 🤝 Citation

```bibtex
@inproceedings{alessi2025hannesimitation,
  title={HannesImitation: Grasping with the Hannes Prosthetic Hand via Imitation Learning},
  author={Alessi, Carlo and Vasile, Federico and Ceola, Federico and Pasquale, Giulia and Boccardo, Nicol{`o} and Natale, Lorenzo},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025},
  pages={},
  organization={}
}
```

### 📬 Contacts
For any query feel free to contact [Carlo Alessi](https://hsp.iit.it/people-details/-/people/carlo-alessi).
