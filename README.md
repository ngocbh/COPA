# Counterfactual plans under distributional ambiguity

This repo contains source-code of the "Counterfactual plans under distributional ambiguity" paper

## Experiments

### Evaluation of Mahalanobis Correction
This experiment evaluates `MahalanobisCrr` on different number of used corrections `K` and perturbation limit `\Delta`.
To run this experiment:

```
python run_epts.py --ept 1 --datasets german sba student --classifiers lrt -uc --run-id <run-id>
```

The results saved in `results/run_<run_id>/ept_1`

### Evaluation on synthetic dataset
This experiment investigates the impact of degree of distribution shift on the validity of a plan.
To run this experiment:

```
python run_epts.py --ept 3 --datasets synthesis --methods dice mahalanobis pgd --classifiers lrt -uc --run-id <run-id>
```

Results saved in `results/run_<run_id>/ept_3`

### Evaluation on real-world datasets
This experiment compares three method `DiCE, MahalanobisCrr, DroDicePGD` in the three real world datasets: `german, sba, student`

1. First, prepare an underlying classifier and 'future' classifiers and for each dataset:

```
python run_epts.py --ept pretrain --datasets german german_shift sba sba_shift student student_shift --classifiers lrt --run-id 0
mv results/run_0/ept_pretrain/*.pickle data/pretrain 
```

2. Run the experiment:
```
python run_epts.py --ept 2 --classifiers lrt --datasets german sba student --methods dice mahalanobis pgd --run-id <run-id> --num-proc 32
```

3. The result is saved in `results/run_<run_id>/ept_2`
