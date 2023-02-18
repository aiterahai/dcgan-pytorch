# dcgan-pytorch

---

This is an implementaiton of DCGAN, as introduced in the paper "[UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)" by Alec Radford et al.

## Prerequisites

---

- python>=3.9.12

- torch==1.13.1

- torchvision==0.14.1

You can install required packages by:

```bash
pip3 install -r requirements.txt
```

## Training

```bash
python3 train.py
```

Specify `TRAIN_IMG_DIR` in the script before training.

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.
