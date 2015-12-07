# Whitening

- Many methods work best after the data has been normalized and whitened.
- A standard pipeline for preprocessing data consists of:
	1. Normalilze data
		- Simple rescaling
		- Per-example mean substraction
		- Feature Standardization
			- move data to have zero mean and unit variance
			- Best thing to do when using linear classifiers
			- First compute the mean of each dimension then divide by the standard deviation of each dimension
	2. Whiten data
		- PCA
		- ZCA


### **Tips and tricks:**

- For color images it is recommended to normalize the color
- Obtain features
- Apply dimension mean normalization
- Calculate the eigenvectors and eigenvalues through SVD
- Graph the eigenvalue vs order 
- Select the epsilon
- Apply ZCA with determined epsilon (if no previous step apply an epsilon of 0.1 or 0.01)

**Parameters to keep:**
1. Feature means
2. Feature standard deviation
3. U and S matrices 


### Plan of action

1. Normalize the colors per sample
2. Whiten
	- Zero mean the features
	- Select epsilon (regularization constant)
		- Choose epsilon that leaves out the longer tail
		- Graph eigenvalue vs eigenvalues order
		- If data has been scaled reasonably start with epsilon = 0.1 or 0.01
	
