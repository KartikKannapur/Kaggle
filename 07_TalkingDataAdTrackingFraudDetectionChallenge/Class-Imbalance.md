# Solving the Class Imbalance Problem

If the dataset used is imbalanced, it will have an impact on the difficulty of getting a good predictive and meaningful model due to the lack of information from the minority class. This standard method will produce a bias toward the classes with a greater number of instances (the majority) because the classifier will tend to predict the majority class data. The minority class will be ignored (treating them as a noise), so the observation from the minority class cannot be classified correctly.

### What are the options out there and why do (or don't) they work?

* __Undersampling__

This technique randomly discards a subset of the majority class until the ratio of the two classes is ~1. While the approach is simple it has had an issue with affecting the underlying distribution of the data and discarding important information that the discarded samples contributed.

_There are variants of undersampling that represent undersampling generalization such as Exactly Balanced Bagging (EBBag), Roughly Balanced Bagging (RBBag). But we will explore them at a later stage._

* __Oversampling__

Oversampling creates multiple copies of the minority class until the ratio between the two classes is ~1. However, while a simple technique, is it susceptible (more often than not) to create an overfit. This is due to the fact that oversampling creates exact replicas of the minority class samples and tightens the decision boundary.

Ideally we do not want to undersample and lose information and we do not want to oversample and overfit the model. Here are some other approaches to avoid this situation.

* __SMOTE__

SMOTE (Synthetic Minority Over-sampling TEchnique) uses a modified oversampling technique. The idea is not to replicate the minority class samples but create synthetic samples using the minority class samples. This offers two advantages:

1. Syntheic samples are similar but not identical to the minority class samples.
2. As a result of the above point, the decision boundary between the two classes is more generalized reducing the chance of overfit.

__*Basic algorithm (Chawla et al.[1])*__:

_(If N is less than 100%, randomize the minority class samples as only a random percent of them will be SMOTEd.)_  
__if__ N<100:  
&nbsp;&nbsp;&nbsp;&nbsp;__then__ Randomize the T minority class samples   
&nbsp;&nbsp;&nbsp;&nbsp;T = (N/100) ∗ T  
&nbsp;&nbsp;&nbsp;&nbsp;N = 100  
__endif__  

_N_ = (int)(N/100) _(The amount of SMOTE is assumed to be in integral multiples of 100.)_  
_k_ = Number of nearest neighbors  
_numattrs_ = Number of attributes  
_Sample[ ][ ]_: array for original minority class samples  
_newindex_: keeps a count of number of synthetic samples generated, initialized to 0  
_Synthetic[ ][ ]_: array for synthetic samples   

_(Compute k nearest neighbors for each minority class sample only.)_   
__for__ i ← 1 __to__ T:  
&nbsp;&nbsp;&nbsp;&nbsp;_(Compute k nearest neighbors for i, and save the indices in the nnarray_  
&nbsp;&nbsp;&nbsp;&nbsp;_Populate(N, i, nnarray)_  
__endfor__    


_(Function to generate the synthetic samples.)_  
__Populate(N, i, nnarray)__:   

&nbsp;&nbsp;&nbsp;&nbsp;__while__ _N != 0_:  
&nbsp;&nbsp;&nbsp;&nbsp;_(Choose a random number between 1 and k, call it nn. This step chooses one of the k nearest neighbors of i.)_  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;__for__ attr ← 1 to numattrs:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute: _dif = Sample[nnarray[nn]][attr] − Sample[i][attr]_     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute: _gap = random number between 0 and 1_  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_Synthetic[newindex][attr] = Sample[i][attr] + gap ∗ dif_  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;__endfor__  

&nbsp;&nbsp;&nbsp;&nbsp;_newindex++_  
&nbsp;&nbsp;&nbsp;&nbsp;_N = N−1_   
&nbsp;&nbsp;&nbsp;&nbsp;__endwhile__    
__return__ (_End of Populate._)  

End of Pseudo-Code.  

We will be considering a version of SMOTE that uses Bagging (SMOTE-Bagging).

## References

[1] V.N. Chawla, K.W. Bowyer, L.O. Hall, W.P. Kegelmeyer, SMOTE: Synthetic Minority Over-Sampling Technique, Journal of Artificial Intelligence Research, 16 (2002), 321-357.
