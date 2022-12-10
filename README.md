# Robust Knowledge Distillation with non-robust Teacher  (PyTorch)


# env setup
```
conda create -n advkd python=3.9
conda activate advkd
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia 
pip install torchattacks argparse scipy 
pip install labml-helpers
```


# Normal Training
```
python xTrain.py --type nt
```

# Adversarial Training
```
python xTrain.py --type at
```

# Knowledge Distillation Training
```
python xTrain.py --type kd
```


# TESTING (Test with flag --attack  PGD / auto )
```
python test.py --filename NT_.pth --attack PGD
```


#Results
![image](https://user-images.githubusercontent.com/114927281/206832113-fadd766a-3562-49fa-b806-04772f3c0c20.png)
