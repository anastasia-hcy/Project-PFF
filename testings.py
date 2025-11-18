#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM"
pathdat             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/data"
pathfig             = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM/plots"

# path                = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/scripts"

import os, sys
os.chdir(path)
cwd = os.getcwd()
print(f"Current working directory is: {cwd}")
sys.path.append(cwd)

############# 
# Libraries #
#############

import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import unittest

##############
# Unit tests #
##############

from functions import SE_kernel, SE_kernel_divC, SE_Cov_div

class TestSE(unittest.TestCase):
    
    def setUp(self):
        self.nD         = 3
        self.nT         = 5
        self.scale      = 2.0 
        self.length     = 1.0 
        self.x          = tf.constant([0.0, 1.0, 2.0], dtype=tf.float64)

    def test_se_kernel(self):
        result          = SE_kernel(self.x[1], self.x[2], self.scale, self.length).numpy()
        expected_result = ((self.scale ** 2.0) * tf.math.exp(- 0.5)).numpy()
        self.assertAlmostEqual(result, expected_result, places=6)
        
    def test_se_kernel_divC(self):
        result          = SE_kernel_divC(self.x[1], self.x[2], self.length).numpy()
        expected_result = 1.0 / self.length 
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_se_cov_div(self):
        M, Md           = SE_Cov_div(self.nD, self.x, self.scale, self.length)
        self.assertEqual(M.shape, (self.nD, self.nD))
        self.assertEqual(Md.shape, (self.nD, self.nD))
        self.assertTrue(tf.reduce_all(M.numpy() == M.numpy().T)) 
        self.assertTrue(tf.reduce_all(Md.numpy() == -Md.numpy().T))  
        self.assertTrue(isinstance(M, tf.Variable))
        self.assertTrue(isinstance(Md, tf.Variable))

from functions import norm_rvs, LGSSM, SV_transform, SVSSM

class TestSSM(unittest.TestCase):
    
    def setUp(self):
        self.nD         = 3
        self.nT         = 5
        self.x          = tf.constant([0.0, 1.0, 2.0], dtype=tf.float64)
        self.mean       = tf.constant([0.0, 0.0, 0.0], dtype=tf.float64)
        self.Sigma      = tf.eye(self.nD, dtype=tf.float64)

    def test_norm_rvs(self):
        sample          = norm_rvs(self.nD, self.mean, self.Sigma)
        self.assertEqual(sample.shape, (self.nD,))
        self.assertTrue(isinstance(sample, tf.Tensor))

    def test_lgssm(self):
        X, Y            = LGSSM(self.nT, self.nD)
        self.assertEqual(X.shape, (self.nT, self.nD))
        self.assertEqual(Y.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X, tf.Variable))
        self.assertTrue(isinstance(Y, tf.Variable))

    def test_sv_transform(self):
        B               = tf.eye(self.nD, dtype=tf.float64)
        W               = tf.eye(self.nD, dtype=tf.float64)
        sample          = SV_transform(self.nD, self.mean, B, self.x, W)
        self.assertEqual(sample.shape, (self.nD,))
        self.assertTrue(isinstance(sample, tf.Tensor))
        
    def test_svssm(self):
        X, Y            = SVSSM(self.nT, self.nD)
        self.assertEqual(X.shape, (self.nT, self.nD))
        self.assertEqual(Y.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X, tf.Variable))
        self.assertTrue(isinstance(Y, tf.Variable))


from functions import KF_Predict, KF_Gain, KF_Filter, KalmanFilter

class TestKF(unittest.TestCase):

    def setUp(self):
        self.nD         = 3
        self.nT         = 5
        self.x          = tf.constant(tf.range(3, dtype=tf.float64))
        self.y          = tf.constant(tf.range(self.nD*self.nT, dtype=tf.float64), shape=(5,3))
        self.P          = tf.eye(self.nD, dtype=tf.float64)
        self.A          = tf.eye(self.nD, dtype=tf.float64)
        self.V          = tf.eye(self.nD, dtype=tf.float64)
        self.B          = tf.eye(self.nD, dtype=tf.float64)
        self.W          = tf.eye(self.nD, dtype=tf.float64)

    def test_kf_predict(self):
        x, P            = KF_Predict(self.x, self.P, self.A, self.V)
        expected_x      = tf.linalg.matvec(self.A, self.x)
        expected_P      = self.A @ self.P @ tf.transpose(self.A) + self.V
        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))
        
        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))

    def test_kf_gain(self):
        K               = KF_Gain(self.P, self.B, self.W)
        M               = self.P @ tf.transpose(self.B) 
        Minv            = tf.linalg.inv(self.B @ M + self.W)
        expected_K      = M @ Minv
        self.assertTrue(np.allclose(K.numpy(), expected_K.numpy()))
        self.assertTrue(isinstance(K, tf.Tensor))

    def test_kf_filter(self):
        y_pred          = tf.linalg.matvec(self.B, self.x)
        K               = KF_Gain(self.P, self.B, self.W)
        x, P            = KF_Filter(self.x, self.P, self.y[0,:], y_pred, self.B, K)

        expected_x      = self.x + tf.linalg.matvec(K, self.y[0,:] - y_pred)
        expected_P      = self.P - self.P @ tf.transpose(self.B) @ tf.transpose(K)

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))

        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))
        
    def test_kalman_filter(self):
        
        X_filtered      = KalmanFilter(y=self.y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))



from functions import EKF_Predict, EKF_Jacobi, EKF_Gain, EKF_Filter, ExtendedKalmanFilter

class TestEKF(unittest.TestCase):

    def setUp(self):
        self.nD         = 3
        self.nT         = 5
        self.x          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        self.y          = tf.constant(tf.range(self.nD*self.nT, dtype=tf.float64), shape=(5,3))
        self.P          = tf.eye(self.nD, dtype=tf.float64)
        self.A          = tf.eye(self.nD, dtype=tf.float64) * 0.9
        self.V          = tf.eye(self.nD, dtype=tf.float64)
        self.B          = tf.eye(self.nD, dtype=tf.float64)
        self.W          = tf.eye(self.nD, dtype=tf.float64)
        self.m          = tf.zeros(self.nD, dtype=tf.float64)
        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
    def test_ekf_predict(self):
        x, P            = EKF_Predict(self.x, self.P, self.A, self.V)
        expected_x      = tf.linalg.matvec(self.A, self.x)
        expected_P      = self.A @ self.P @ tf.transpose(self.A) + self.V
        
        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))

    def test_ekf_jacobi(self):
        y_pred          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = EKF_Jacobi(self.x, y_pred, self.B)
        
        expected_Jx     = tf.linalg.diag(y_pred / 2)
        expected_Jw     = tf.linalg.diag(tf.linalg.matvec(self.B, tf.math.exp(self.x / 2)))

        self.assertTrue(np.allclose(Jx.numpy(), expected_Jx.numpy()))
        self.assertTrue(np.allclose(Jw.numpy(), expected_Jw.numpy()))

    def test_ekf_gain(self):
        ypred           = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = EKF_Jacobi(self.x, ypred, self.B)
        K               = EKF_Gain(self.P, Jx, Jw, self.W, self.u)
        
        Mx              = self.P @ tf.transpose(Jx)
        J               = Jx @ Mx + Jw @ self.W @ tf.transpose(Jw)
        Minv            = tf.linalg.inv(J + self.u)
        expected_K      = Mx @ Minv

        self.assertTrue(np.allclose(K.numpy(), expected_K.numpy()))
        self.assertTrue(isinstance(K, tf.Tensor))
        
    def test_ekf_filter(self):
        y_pred          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = EKF_Jacobi(self.x, y_pred, self.B)
        K               = EKF_Gain(self.P, Jx, Jw, self.W, self.u)
        x, P            = EKF_Filter(self.x, self.P, self.y[0, :], y_pred, Jx, K)

        expected_x      = self.x + tf.linalg.matvec(K, self.y[0, :] - y_pred)
        expected_P      = self.P - self.P @ tf.transpose(Jx) @ tf.transpose(K)

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))
        
        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))

    def test_extended_kalman_filter(self):
        X_filtered = ExtendedKalmanFilter(self.y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))









if __name__ == '__main__':
    unittest.main()
    
    
    
    

# nT              = 20
# nD              = 5




# M4 = tf.random.normal((nT,nD), 0, 1, dtype=tf.float64)
# v4 = tf.random.normal((nT,), 0, 1, dtype=tf.float64)
# M5 = tf.Variable(tf.zeros((nT,nD), dtype=tf.float64))
# for i in range(nT):
#     M5[i,:].assign(v4[i] * M4[i,:])

# np.all(
#     np.round(tf.linalg.matvec( tf.transpose(M4), v4), decimals=9) == np.round(tf.reduce_sum(M5, axis=0), decimals=9)
# )


# v1              = tf.random.normal((nD,), 0, 1)
# v2              = tf.random.normal((nD,), 0, 1)
# tf.norm(v1-v2, ord='euclidean')

# M               = tf.Variable(tf.random.normal((nT,nD), 0, 1))
# M2              = tf.Variable(tf.random.normal((nT,nT), 0, 1))
# M3              = tf.Variable(initial_value=tf.zeros((nT,nD)))
# M4              = tf.Variable(initial_value=tf.zeros((nT,nD)))
# for i in range(nD):
#     v3 = tf.Variable(tf.zeros((nT,)))
#     for j in range(nT):
#         for k in range(nT):
#             v3[j].assign_add( M[j,i] * M2[j,k] + M2[j,k]  )
#     M3[:,i].assign_add(v3)
            
# np.all( M3 == M4 )

# tf.Variable( tf.reduce_mean(M, axis=0) )



# from functions import SE_kernel_divC, SE_kernel, SE_Cov_div


# nT = 100000
# (nT**2)/2 + nT < nT**2


# v              = tf.random.normal((nD,), 0, 1, dtype=tf.float64)
# M              = tf.Variable(tf.zeros((nD,nD), dtype=tf.float64))
# M2             = tf.Variable(tf.zeros((nD,nD), dtype=tf.float64))
# for i in range(nD): 
#     for j in range(nD): 
#         M[i,j].assign( SE_kernel(v[i], v[j], 1.0, 1.0) )
#     for j in range(i,nD):
#         M2[i,j].assign( SE_kernel(v[i], v[j], 1.0, 1.0) )
#         M2[j,i].assign( SE_kernel(v[i], v[j], 1.0, 1.0) )
         
# M3, M4          = SE_Cov_div(nD, v)
# np.all( M == M2 )
# np.all( M == M3 )





# from functions import KPFF_RKHS

# v              = tf.random.normal((nT,), 0, 1, dtype=tf.float64)
# v2             = tf.random.normal((nT,), 0, 1, dtype=tf.float64)
# M              = tf.Variable(tf.zeros((nT,nD), dtype=tf.float64))
# M2             = tf.Variable(tf.zeros((nT,nD), dtype=tf.float64))
# for i in range(nD): 
#     K, Kd       = SE_Cov_div(nT, v)
#     for j in range(nT): 
#         M[j,i].assign( tf.reduce_sum( 1/nT * (v2[j] * K[j,:] + Kd[j,:])) )
#     for j in range(nT):
#         val = 0.0
#         for k in range(nT):
#             Kj   = SE_kernel(v[j], v[k])
#             Kdj  = SE_kernel_div(v[j], v[k], length=1.0) 
#             val += 1/nT * ( v2[j] * Kj + Kdj) 
#         M2[j,i].assign(val) 

# np.all( 
#     np.round(M, decimals=9) == np.round(M2, decimals=9)
# )


# tf.math.multiply(v1, M4)




# M              = tf.Variable(tf.random.normal((nT,nD), dtype=tf.float32))
# tf.reduce_sum(M, axis=1)

# M              = tf.Variable(tf.random.normal((nD,nD), dtype=tf.float32))
# v              = tf.random.normal((nD,), 0, 1)
# v2             = tf.linalg.matvec(M, v)
# M3             = tf.Variable(tf.zeros((nD,nD), dtype=tf.float32))
# for i in range(nD):
#     for j in range(nD): 
#         M3[i,j].assign(v[i] * M[i,j] * v[j])
        
# np.all(
#    M3 == tf.linalg.diag(v) @ M @ tf.linalg.diag(v)
# ) 

# np.all( 
#     sum(v*v) == tf.linalg.tensordot(v, v, axes=1) 
# )

# tf.linalg.tensordot(v, v, axes=0) 


# tf.range(0.0, 1.0/1.25, 1e-3, dtype=tf.float64) * 1.25

# v3 = tf.random.normal((nD,), 0, 1)
# np.all( 
#     tf.linalg.diag_part( 
#         tf.linalg.tensordot(tf.linalg.matvec(M, v3 ), v3, axes=0) 
#     ) == tf.linalg.diag_part( v3 * M * v3) 
# )

# np.all( 
#     tf.linalg.diag_part(M) * (v3**2) == tf.linalg.diag_part( v3 * M * v3) 
# )




# from functions import norm_rvs

# np.all(
#     tf.linalg.matvec(tf.linalg.diag([1.0,2.0,3.0]), tf.ones((3,))) + tf.ones((3,)) == tf.constant([2.0, 3.0, 4.0], dtype=tf.float32)
# )

# norm_rvs(1, tf.zeros((1,)), tf.eye(1))

# x0 = norm_rvs(nT, tf.zeros((nT,)), tf.eye(nT)) 
# y0 = norm_rvs(nT, x0, tf.eye(nT))

# plt.plot(x0, linewidth=1, alpha=0.75) 
# plt.plot(y0, linewidth=1, alpha=0.75) 
# plt.show() 
 






# from functions import LGSSM

# v       = tf.random.normal((nD,), 0, 1)
# A       = tf.random.normal((nD,nD), 0, 1)
# B       = tf.random.normal((nD,nD), 0, 1)
# np.all( 
#     A @ B @ tf.transpose(A) == tf.matmul( tf.matmul(A,B), A, transpose_b=True) 
# )
# np.all(
#     v * A * tf.transpose(v) == tf.linalg.matvec(A, v) * v
# )






# from functions import SVSSM

# v2      = tf.random.normal((nD,), 0, 1)
# np.all( 
#     v * tf.eye(nD) * v == tf.linalg.diag(v**2) 
# )
# np.all(
#     tf.linalg.matvec(B, v) * (tf.eye(nD)*v2) * tf.linalg.matvec(B, v) == tf.linalg.diag(tf.linalg.matvec(B, v) * v2 * tf.linalg.matvec(B, v))
# ) 



# from functions import SigmaPoints, Scaling, SigmaWeights

# A = tf.random.normal((nD,nD), 0, 1)
# SP = SigmaPoints(nD, X[0,:], tf.eye(nD), tf.eye(nD), 1.0)

# M = SP[:,:nD] @ tf.transpose(A)
# M.shape
# for i in range(2*nD+1):
#     if np.all(
#         np.round(M[i,:], decimals=5) == np.round(tf.linalg.matvec(A, SP[i,:nD]), decimals=5)
#     ) == False:
#         print("Mismatch at index ", i)
#         print(M[i,:])
#         print(tf.linalg.matvec(A, SP[i,:nD]) )
        
# sw = SigmaWeights(nD)  
        









# class ClassA:
#     def method_in_a(self):
#         print("Method from ClassA called.")

# class ClassB:
#     def call_method_from_a(self):
#         # Create an instance of ClassA
#         instance_of_a = ClassA()
#         # Call the method using the instance
#         instance_of_a.method_in_a()

# # Example usage:
# b_obj = ClassB()
# b_obj.call_method_from_a()