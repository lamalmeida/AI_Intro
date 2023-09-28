import numpy as np
from scipy.sparse import csr_matrix
import time

def power_iter(X, num_iter:int):
  v = np.random.randn(X.shape[1]) 
  one_vec = np.ones(X.shape[0]) 
  mu_row_matrix = np.mean(X, axis=0)  
  mu = np.array(mu_row_matrix).squeeze()
  for _ in range(num_iter):
    v = np.transpose(X).dot(X.dot(v))-mu.dot(np.transpose(one_vec).dot(X.dot(v)))-(np.transpose(X).dot(one_vec)).dot(np.transpose(mu).dot(v))+np.dot(mu,np.dot(np.dot(np.transpose(one_vec),one_vec),np.dot(np.transpose(mu),v)))
    v_norm = np.linalg.norm(v,ord=2)
    v = v / v_norm
  return v

def verify_v1(X):
  X = X - np.mean(X, axis=0)
  _,_,X = np.linalg.svd(X)
  return X[0,:]

if __name__ == '__main__':
   for threshold in [0.9, 0.99, 0.999, 0.9999]:
    print(f"Nonzero {(1 - threshold) * 100:.2f}%")
    for dim in [100, 1000, 10000]:
        X = np.random.uniform(0,1,dim*dim)
        X = X.reshape(dim,dim)
        X[X < threshold] = 0
        X_sparse = csr_matrix(X)

        t_start = time.perf_counter()
        power_iter(X,10)
        t_end = time.perf_counter()
        normal_time = t_end - t_start
        t_start = time.perf_counter()
        power_iter(X_sparse,10)
        t_end = time.perf_counter()
        sparse_time = t_end - t_start
        ratio = normal_time/sparse_time
        print(f"Size: {dim}x{dim} - Sparse method is {ratio if ratio > 1 else 1/ratio:.3f} times {'faster' if ratio >1 else 'slower'}")
