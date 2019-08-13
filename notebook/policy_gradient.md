## Derivation of Jacobian w.r.t. Model Parameters

**Definition 1.1:** The tensor dot product is a sum reduction along the last and first dimension of the two operands respectively, i.e. $\sf {\bf C} = a_{...n} \ b_{n...}$.

**Definition 1.2:** The jacobian evaluated at a given instance maps a function to the tensor product of its codomain and domain, i.e. $\sf \grad[\bf V] : {\bf W} \rightarrow {\bf W} \otimes {\bf V}$.

**Proof:** Let $\sf {\bf A} \in \reals^{m \times m}$, $\sf {\bf v} \in \reals^{m}$, and $\sf {\cal f}: \ \reals \rightarrow \reals$.
$$
\begin{align}
    \sf v_{n} &\sf = {\cal f} ({\bf A}^{\top}_{*n} \ {\bf v}) \\
    \sf \grad[\bf A] v_{n} &\sf = \grad[\bf A] {\cal f} ({\bf A}^{\top}_{*n} \ {\bf v}) \tag{1}
\end{align}
$$
To evaluate the jacobian of $\cal f$ w.r.t. matrix $\bf A$, the partial derivatives w.r.t. each element of $\bf A$ are deduced under Einstein's summation notation.
$$
\begin{align}
    \sf \pdv[v_{n}]{a_{pq}} = \pdv[\cal f]{a_{pq}} (a_{kn} \ v_{k}) &\sf = (\grad {\cal f}) (a_{kn} \ v_{k}) \ \pdv{a_{pq}} [a_{kn} \ v_{k}] \\
    &\sf = (\grad {\cal f}) ({\bf A}^{\top}_{*n} \ {\bf v}) \ [\pdv[a_{kn}]{a_{pq}} \ v_{k} + a_{kn} \ \pdv[v_{k}]{a_{pq}}] \\
    &\sf = (\grad {\cal f}) ({\bf A}^{\top}_{*n} \ {\bf v}) \ [\delta_{qn} \ v_{p} + {\bf A}^{\top}_{*n} \ (\grad[\bf A] {\bf v})_{pq}] \tag{2}
\end{align}
$$
The individual derivatives are then compiled into a jacobian constructed under matrix notation.
$$
\begin{align}
    \sf (\grad[\bf A] v_{n})_{*q} &\sf = (\grad {\cal f}) ({\bf A}^{\top}_{*n} \ {\bf v}) \ [\delta_{qn} \ {\bf v}^{\top} + {\bf A}^{\top}_{*n} \ (\grad[\bf A] {\bf v})_{*q}]  \\
    \sf \therefore \grad[\bf A] v_{n} &\sf = (\grad {\cal f}) ({\bf A}^{\top}_{*n} \ {\bf v}) \ [1 \minify{\otimes} ({\bf v} \ \delta^{\top}_{*n}) + {\bf A}^{\top}_{*n} \ \grad[\bf A] {\bf v}]  \tag{3}
\end{align}
$$

