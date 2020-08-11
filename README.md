# AdaCorr

This is the code for the paper:

**<a> Error-Bounded Correction of Noisy Labels </a>**
<br>
Songzhu Zheng, Pengxiang Wu, Aman Goswami, Mayank Goswami, Dimitris Metaxas, Chao Chen
</br>
Presented at [ICML 2020](https://icml.cc/Conferences/2020)

If you find this code useful in your research please cite:
```
@inproceedings{jiang2018mentornet,
  title={Error-Bounded Correction of Noisy Labels},
  author={Songzhu Zheng, Pengxiang Wu, Aman Goswami, Mayank Goswami, Dimitris Metaxas, Chao Chen},
  booktitle={ICML},
  year={2020}
}
```

## Introduction 

AdaCorr is an algorithm that is designed to perform robustly when label noise is presented in your training data. This algorithm will try to correct mislabeled data points during training process relying on current network's confidence. 

Specifically, for each input data point
- (1) Likelihood ratio test (LRT) is performed to test the correctness of its current label. If the label is rejected by the test, the algorithm will set the new label to be the current MLE. 
- (2) Network will keep trainning using the new labels, which refines the power of the test

Steps (1) and (2) are conducted iteratively until the learning procedure converges.

## Environment Setting

## Running AdaCorr 

## Parameter Setting for Experiment Result 

## Performance

## Algorithm

## Reference

