# Fundamentals of Graph Learning

## Graph Fourier Transform

Given a graph $G=(V,E)$, we have its adjacency matrix $A$. We denote the Laplacian matrix $L$ as $L=D-A$, where $D$ is the diagonal degree matrix. One important property of the Laplacian matrix is as follows.

Given node feature (1D signal) $x\in \mathbb{R}^n$ where $n=|V|$, we have
$$
x^TLx=\frac{1}{2}\sum_{i,j}a_{ij}(x_i-x_j)^2=\frac{1}{2}\sum_{(i,j)\in E}(x_i-x_j)^2
$$
We can see that $L\succeq 0$. From the second equivalence, we can see that $x^TLx$ can represent the frequency of the graph, *i.e.* how much signal varies from node to node. Then we try to find a set of basis like $e^{j\omega}$ in Fourier Transform of signals. Denote the basis as $U$, then we have $L=U^T\Sigma U^T$, where $\Sigma=\text{diag}(\lambda_1,\dots,\lambda_n)$.

Apparently, we have $Lu_i=\lambda_iu_i$, thus we have $u_i^TLu_i=\lambda_i$. From the previous deduction we can see that $\lambda_i$ is the frequency. Following common practice, we usually require $\lambda_1\le \lambda_2\dots \le\lambda_n$, representing low-frequency to high-frequency signals.

Since $U$ is a set of basis in $\mathbb{R}^n$, we can naturally decompose it with respect to $U$. Thus, we have
$$
x=\sum_{i=1}^n \beta_iu_i\Rightarrow u_j^Tx=\sum_{i=1}^n \beta_iu_j^Tu_i
$$
Since $\langle u_i,u_j\rangle=1$ iff $u_i=u_j$ due to property of orthogonal vectors, we have $u_j^Tx=\beta_j$. Thus, the Graph Fourier Transform (GFT) and the inverse transform (IGFT) can be represented as 
$$
\tilde{X}=U^TX
$$

$$
X=U^{-T}\tilde{X}=U\tilde{X}
$$

## Graph Signal Processing

Consider the following, where $g(\cdot)$ is a signal processing function (filter). 
$$
[u_1,\dots, u_n]g(\Sigma)[u_1^Tx, \dots, u_n^Tx]^T
$$
For example, a full-band filter is given by $g(\Sigma)=I$. Then we have $uIu^Tx=uu^Tx=x$.

If we use the the Laplacian matrix $L$, *i.e.* $g(\Sigma)=\Sigma$, then we can see that we are suppressing the low frequency signals and amplifying the high frequency signals. So we can view $L$ as a high-frequency filter (only high frequency signals are passed). In reality, we need to normalize $L$ to prevent signal explosion by using $D^{-1}L=I-D^{-1}A$ instead of $L$. Another normalization method is given by $D^{-\frac{1}{2}}LD^{-\frac{1}{2}}=I-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$. This is more commonly used, *e.g.* in Graph Convolutional Network (GCN). We can also observe that $\tilde{L}=I-\tilde{A}$.

If we use the adjacency matrix $A$, then we have $Ax=U(I-\Sigma)U^Tx$, we can observe that $g(\lambda)=1-\lambda$. So we can consider it as a low-pass filter.

A view of designing GNN is to learn different frequency filters for multiple parts of the signal. Observe that $g(\Sigma)=\sum_i ^k h_i\Sigma^i$, $h_i$ differs.

If we want to merge different layers of information in the final project, we can use $A^i$ to create a new adjacency matrix on which the model can be trained. For example, we can create a new adjacency matrix $\hat{A}=\frac{1}{2}A+\frac{1}{2}A^2$