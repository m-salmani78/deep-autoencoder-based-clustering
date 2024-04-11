# DAC: Deep Autoencoder-based Clustering

## Introduction

This project implements the DAC (Deep Autoencoder-based Clustering) framework as described in the paper titled "DAC: Deep Autoencoder-based Clustering, a General Deep Learning Framework of Representation Learning" by Si Lu and Ruisi Li from Portland State University. DAC is a general deep learning framework designed to enhance clustering performance by leveraging representation learning through deep autoencoders, aimed at addressing high-dimensional data challenges.

**Please note:** This implementation is based on my interpretation of the original paper, and I am not one of the original authors. For detailed insights and theoretical background, please refer to the original publication.

## Features

- **Generalized Framework**: Suitable for various types of datasets including images and sensor data.
- **Enhanced Clustering**: Significantly boosts the performance of K-Means clustering algorithm.
- **Customizable**: Easily adaptable to different datasets and clustering algorithms.

## Datasets

The DAC framework has been tested on the following datasets:
- MNIST (hand-written digits)
- Fashion-MNIST (clothing images)
- HAPT (Human Activities and Postural Transitions Data Set)

## Results

Our approach has shown to significantly improve the clustering performance across various datasets. For instance, on the MNIST dataset, DAC increased the Adjusted Rand Index (ARI) from 0.3477 to 0.6624.

## Contributing

We welcome contributions to improve the DAC framework! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## Citation and Acknowledgment

If you find this implementation useful in your research, please consider citing the original paper:

```bibtex
@article{lu2021dac,
  title={DAC: Deep Autoencoder-based Clustering, a General Deep Learning Framework of Representation Learning},
  author={Si Lu and Ruisi Li},
  journal={arXiv preprint arXiv:2102.07472},
  year={2021}
}
```

For the original research and paper, please refer to the [arXiv link](https://arxiv.org/abs/2102.07472) (if freely available) or search for the title "DAC: Deep Autoencoder-based Clustering, a General Deep Learning Framework of Representation Learning" in academic databases.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

