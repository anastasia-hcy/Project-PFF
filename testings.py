#################
# Set directory #
#################

# path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM"
path                = "C:/Users/CSRP.CSRP-PC13/Projects/Practice/scripts/"

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
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.random.set_seed(123)

import math
pi_constant = tf.constant(math.pi, dtype=tf.float64)

import unittest
from functions import SE_kernel, SE_kernel_divC, SE_Cov_div
from functions import norm_rvs, LG_Jacobi
from functions import SV_transform, SV_Jacobi
from functions import measurements, measurements_Jacobi, measurements_covyHat, SSM

from functions import KF_Predict, KF_Gain, KF_Filter, KalmanFilter
from functions import EKF_Predict, EKF_Gain, EKF_Filter, ExtendedKalmanFilter
from functions import SigmaWeights, SigmaPoints, UKF_Propagate, UKF_Predict_mean, UKF_Predict_cov, UKF_Predict_crosscov, UKF_Predict_additive, UKF_Predict_augmented, UKF_Gain, UKF_Filter, UnscentedKalmanFilter

from functions import initiate_particles, draw_particles, LogImportance, LogLikelihood, LogTarget, compute_weights, normalize_weights, compute_ESS, multinomial_resample, compute_posterior
from functions import ParticleFilter

from functions import Li17eq10, Li17eq11
from functions import EDH_linearize_EKF, EDH_linearize_UKF, EDH_flow_dynamics, EDH_flow_lp, EDH
from functions import LEDH_linearize_EKF, LEDH_linearize_UKF, LEDH_flow_dynamics, LEDH_flow_lp, LEDH
from functions import Hu21eq13, Hu21eq15, KPFF_LP, KPFF_RKHS, KPFF_flow, KernelPFF

##############
# Unit tests #
##############

class TestSE(unittest.TestCase):
    
    def setUp(self):
        self.nD         = 30
        self.scale      = 2.0 
        self.length     = 1.0 
        self.X          = tf.constant(tf.range(self.nD, dtype=tf.float64))

    def test_se_kernel(self):
        result          = SE_kernel(self.X[1], self.X[2], self.scale, self.length).numpy()
        expected_result = ((self.scale ** 2.0) * tf.math.exp(- 0.5)).numpy()
        self.assertAlmostEqual(result, expected_result, places=6)
        
    def test_se_kernel_divC(self):
        result          = SE_kernel_divC(self.X[1], self.X[2], self.length).numpy()
        expected_result = 1.0 / self.length 
        self.assertAlmostEqual(result, expected_result, places=6)

    def test_se_cov_div(self):

        M, Md           = SE_Cov_div(self.nD, self.X, self.scale, self.length)
        M2              = tf.Variable(tf.zeros((self.nD,self.nD), dtype=tf.float64))
        Md2             = tf.Variable(tf.zeros((self.nD,self.nD), dtype=tf.float64))
        for i in range(self.nD): 
            for j in range(self.nD):
                sij     = SE_kernel(self.X[i], self.X[j], self.scale, self.length) 
                sdij    = SE_kernel_divC(self.X[i], self.X[j], self.length) 
                M2[i,j].assign(sij)
                Md2[i,j].assign(sdij)

        self.assertEqual(M.shape, (self.nD, self.nD))
        self.assertEqual(Md.shape, (self.nD, self.nD))

        self.assertTrue(np.allclose(M.numpy(), M2.numpy()))
        self.assertTrue(np.allclose(Md.numpy(), Md2.numpy()))
        self.assertTrue(tf.reduce_all(M.numpy() == M.numpy().T)) 
        self.assertTrue(tf.reduce_all(Md.numpy() == - Md.numpy().T))  

        self.assertTrue(isinstance(M, tf.Variable))
        self.assertTrue(isinstance(Md, tf.Variable))


class TestSSM(unittest.TestCase):
    
    def setUp(self):

        self.nD         = 3
        self.nO         = self.nD - 1 
        self.nT         = 10

        self.X          = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nO,), dtype=tf.float64)

        self.mean       = tf.zeros((self.nO,), dtype=tf.float64)
        self.Sigma      = tf.eye(self.nO, dtype=tf.float64)

        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.random.uniform((self.nO,self.nD), -10.0, 10.0, dtype=tf.float64)
        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        
        self.W          = tf.linalg.diag(tf.random.uniform((self.nO,), 1e-3, 2.0, dtype=tf.float64))
        self.I          = tf.ones((self.nO,), dtype=tf.float64)

    def test_norm_rvs(self):
        sample          = norm_rvs(self.nO, self.mean, self.Sigma)

        self.assertEqual(sample.shape, (self.nO,))
        self.assertTrue(isinstance(sample, tf.Tensor))

    # def test_lgssm(self):
    #     X, Y            = LGSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)

    #     self.assertEqual(X.shape, (self.nT, self.nD))
    #     self.assertEqual(Y.shape, (self.nT, self.nD))

    #     self.assertTrue(isinstance(X, tf.Variable))
    #     self.assertTrue(isinstance(Y, tf.Variable))

    def test_lg_jacobi(self):

        Jx, Jw          = LG_Jacobi(self.Y, self.B)

        expected_Jx     = tf.linalg.diag( tf.reduce_sum(self.B, axis=1) )
        expected_Jw     = tf.linalg.diag( self.Y )

        self.assertEqual(Jx.shape, (self.nO, self.nO))
        self.assertEqual(Jw.shape, (self.nO, self.nO))

        self.assertTrue(np.allclose(Jx.numpy(), expected_Jx.numpy()))
        self.assertTrue(np.allclose(Jw.numpy(), expected_Jw.numpy()))

    def test_sv_transform(self):
        
        u               = tf.eye(self.nO, dtype=tf.float64)
        sample          = SV_transform(self.nO, self.mean, self.B, self.X, self.W, u)

        self.assertEqual(sample.shape, (self.nO,))
        self.assertTrue(isinstance(sample, tf.Tensor))

    def test_SV_Jacobi(self):

        y_pred          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = SV_Jacobi(self.X, y_pred, self.B)

        expected_Jx     = tf.linalg.diag(y_pred / 2)
        expected_Jw     = tf.linalg.diag(tf.linalg.matvec(self.B, tf.math.exp(self.X / 2)))

        self.assertEqual(Jx.shape, (self.nD, self.nD))
        self.assertEqual(Jw.shape, (self.nO, self.nO))

        self.assertTrue(np.allclose(Jx.numpy(), expected_Jx.numpy()))
        self.assertTrue(np.allclose(Jw.numpy(), expected_Jw.numpy()))
        
    def test_measurements(self):

        ylg             = measurements("LG", self.nO, self.mean, self.B, self.X, self.W)
        ysv             = measurements("SV", self.nO, self.mean, self.B, self.X, self.W)

        self.assertEqual(ylg.shape, (self.nO,))
        self.assertEqual(ysv.shape, (self.nO,))

        self.assertTrue(isinstance(ylg, tf.Tensor))
        self.assertTrue(isinstance(ysv, tf.Tensor))

    def test_measurements_jacobi(self):

        JxLG, JwLG      = measurements_Jacobi("LG", self.I, self.X, self.Y, self.B)
        JxSV, JwSV      = measurements_Jacobi("SV", self.I, self.X, self.Y, self.B)
        
        expected_JxLG   = tf.linalg.diag(tf.reduce_sum(self.B, axis=1))
        expected_JwLG   = tf.linalg.diag(self.I)

        expected_JxSV   = tf.linalg.diag(self.Y / 2)
        expected_JwSV   = tf.linalg.diag(tf.linalg.matvec(self.B, tf.math.exp(self.X / 2)))

        self.assertEqual(JxLG.shape, (self.nO,self.nO))
        self.assertEqual(JwLG.shape, (self.nO,self.nO))

        self.assertEqual(JxSV.shape, (self.nO,self.nO))
        self.assertEqual(JwSV.shape, (self.nO,self.nO))

        self.assertTrue(isinstance(JxLG, tf.Tensor))
        self.assertTrue(isinstance(JwLG, tf.Tensor))
        self.assertTrue(isinstance(JxSV, tf.Tensor))
        self.assertTrue(isinstance(JwSV, tf.Tensor))
        
        self.assertTrue(np.allclose(JxLG.numpy(), expected_JxLG.numpy()))
        self.assertTrue(np.allclose(JwLG.numpy(), expected_JwLG.numpy()))
        self.assertTrue(np.allclose(JxSV.numpy(), expected_JxSV.numpy()))
        self.assertTrue(np.allclose(JwSV.numpy(), expected_JwSV.numpy()))
        
    def test_measurments_cov(self):

        covLG           = measurements_covyHat("LG", self.nO, tf.linalg.diag(self.I), self.W)
        covSV           = measurements_covyHat("SV", self.nO, tf.linalg.diag(self.I), self.W, tf.eye(self.nO, dtype=tf.float64) * 0.1 )

        expected_covLG = tf.linalg.diag(self.I)
        expected_covSV = tf.linalg.diag(self.I) @ self.W @ tf.linalg.diag(self.I) + tf.eye(self.nO, dtype=tf.float64) * 0.1 

        self.assertEqual(covLG.shape, (self.nO,self.nO))
        self.assertEqual(covSV.shape, (self.nO,self.nO))

        self.assertTrue(np.allclose(covLG.numpy(), expected_covLG.numpy()))
        self.assertTrue(np.allclose(covSV.numpy(), expected_covSV.numpy()))

    def test_ssm(self):

        X, Y            = SSM(self.nT, self.nD, model="SV", A=self.A, V=self.V)

        self.assertEqual(X.shape, (self.nT, self.nD))
        self.assertEqual(Y.shape, (self.nT, self.nD))

        self.assertTrue(isinstance(X, tf.Variable))
        self.assertTrue(isinstance(Y, tf.Variable))

        XS, YS          = SSM(self.nT, self.nD, n_sparse=self.nO, model="SV", A=self.A, V=self.V)

        self.assertEqual(XS.shape, (self.nT, self.nD))
        self.assertEqual(YS.shape, (self.nT, self.nO))

        self.assertTrue(isinstance(XS, tf.Variable))
        self.assertTrue(isinstance(YS, tf.Variable))
        
        X1, Y1          = SSM(self.nT, self.nD, A=self.A, V=self.V)

        self.assertEqual(X1.shape, (self.nT, self.nD))
        self.assertEqual(Y1.shape, (self.nT, self.nD))

        self.assertTrue(isinstance(X, tf.Variable))
        self.assertTrue(isinstance(Y, tf.Variable))

        XS2, YS2        = SSM(self.nT, self.nD, n_sparse=self.nO, A=self.A, B=self.B, V=self.V)

        self.assertEqual(XS2.shape, (self.nT, self.nD))
        self.assertEqual(YS2.shape, (self.nT, self.nO))

        self.assertTrue(isinstance(XS2, tf.Variable))
        self.assertTrue(isinstance(YS2, tf.Variable))


class TestKF(unittest.TestCase):

    def setUp(self):

        self.nD         = 3
        self.nT         = 10
        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)

        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64)) 
        
        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64)) 
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

    def test_kf_predict(self):

        x, P            = KF_Predict(self.Xprev, self.P, self.A, self.V)
        expected_x      = tf.linalg.matvec(self.A, self.Xprev)
        expected_P      = self.A @ self.P @ tf.transpose(self.A) + self.V

        self.assertEqual(x.shape, (self.nD,))
        self.assertEqual(P.shape, (self.nD, self.nD))

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))
        
        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))

    def test_kf_gain(self):

        K               = KF_Gain(self.P, self.B, self.W)
        M               = self.P @ tf.transpose(self.B) 
        Minv            = tf.linalg.inv(self.B @ M + self.W)
        expected_K      = M @ Minv

        self.assertEqual(K.shape, (self.nD, self.nD))
        self.assertTrue(np.allclose(K.numpy(), expected_K.numpy()))
        self.assertTrue(isinstance(K, tf.Tensor))

    def test_kf_filter(self):

        y_pred          = tf.linalg.matvec(self.B, self.Xprev)
        K               = KF_Gain(self.P, self.B, self.W)
        x, P            = KF_Filter(self.Xprev, self.P, self.Y, y_pred, self.B, K)

        expected_x      = self.Xprev + tf.linalg.matvec(K, self.Y - y_pred)
        expected_P      = self.P - self.P @ tf.transpose(self.B) @ tf.transpose(K)

        self.assertEqual(x.shape, (self.nD,))
        self.assertEqual(P.shape, (self.nD, self.nD))

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))

        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))
        
    def test_kalman_filter(self):
        
        _, Y            = SSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered      = KalmanFilter(y=Y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))



class TestEKF(unittest.TestCase):

    def setUp(self):

        self.nD         = 3
        self.nT         = 10
        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.m          = tf.zeros(self.nD, dtype=tf.float64)
        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
    def test_ekf_predict(self):

        x, P            = EKF_Predict(self.Xprev, self.P, self.A, self.V)
        expected_x      = tf.linalg.matvec(self.A, self.Xprev)
        expected_P      = self.A @ self.P @ tf.transpose(self.A) + self.V
        
        self.assertEqual(x.shape, (self.nD,))
        self.assertEqual(P.shape, (self.nD, self.nD))

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))

    def test_ekf_gain(self):

        ypred           = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = SV_Jacobi(self.Xprev, ypred, self.B)
        K               = EKF_Gain(self.P, Jx, Jw, self.W, self.u)
        
        Mx              = self.P @ tf.transpose(Jx)
        J               = Jx @ Mx + Jw @ self.W @ tf.transpose(Jw)
        Minv            = tf.linalg.inv(J + self.u)
        expected_K      = Mx @ Minv

        self.assertTrue(np.allclose(K.numpy(), expected_K.numpy()))
        self.assertTrue(isinstance(K, tf.Tensor))
        
    def test_ekf_filter(self):

        y_pred          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = SV_Jacobi(self.Xprev, y_pred, self.B)
        K               = EKF_Gain(self.P, Jx, Jw, self.W, self.u)
        x, P            = EKF_Filter(self.Xprev, self.P, self.Y, y_pred, Jx, K)

        expected_x      = self.Xprev + tf.linalg.matvec(K, self.Y - y_pred)
        expected_P      = self.P - self.P @ tf.transpose(Jx) @ tf.transpose(K)

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))
        
        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))

    def test_extended_kalman_filter(self):

        _, Y            = SSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered      = ExtendedKalmanFilter(y=Y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))

        _, Y2           = SSM(self.nT, self.nD, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered2     = ExtendedKalmanFilter(y=Y2, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered2.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered2, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))


class TestUKF(unittest.TestCase):

    def setUp(self):
        
        self.nD         = 3
        self.nT         = 10

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.X          = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.m          = tf.zeros(self.nD, dtype=tf.float64)
        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9

        self.alpha      = 1.0
        self.kappa      = 3.0 * self.nD / 2.0
        self.beta       = 2.0
        self.Lambda     = (self.alpha ** 2) * self.kappa

    def test_sigma_weights(self):

        wm, wc, wi, L   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)

        expected_Lambda = (self.alpha ** 2) * self.kappa
        expected_wm     = (expected_Lambda - self.nD) / expected_Lambda
        expected_wc     = expected_wm + (1 - self.alpha ** 2 + self.beta)
        expected_wi     = 1 / (2 * expected_Lambda)

        self.assertAlmostEqual(wm, expected_wm, places=6)
        self.assertAlmostEqual(wc, expected_wc, places=6)
        self.assertAlmostEqual(wi, expected_wi, places=6)
        self.assertAlmostEqual(L, expected_Lambda, places=6)

    def test_sigma_points(self):

        SP              = SigmaPoints(self.nD, self.Xprev, self.P, self.Lambda)
        sqrtMat         = tf.linalg.sqrtm(self.Lambda * self.P)
        
        self.assertEqual(SP.shape, (2 * self.nD + 1, self.nD))
        self.assertTrue(np.allclose(SP[0, :].numpy(), self.Xprev.numpy()))
        for i in range(1, self.nD):
            self.assertTrue(np.allclose( SP[i, :].numpy(), (self.Xprev + sqrtMat[:, i]).numpy() ))
            self.assertTrue(np.allclose( SP[self.nD + i, :].numpy(), (self.Xprev - sqrtMat[:, i]).numpy() ))

    def test_ukf_propagate(self):
        
        x0              = tf.random.uniform((self.nD*2,), 1e-3, 2.0, dtype=tf.float64)
        P0              = tf.linalg.diag(tf.random.uniform((self.nD*2,), 1e-3, 2.0, dtype=tf.float64))
        SP              = SigmaPoints(self.nD*2, x0, P0, self.Lambda)
        
        yspLG           = UKF_Propagate("LG", self.nD, SP, self.B)
        yspSV           = UKF_Propagate("SV", self.nD, SP, self.B)

        expected_LG     = SP[:,:self.nD] @ tf.transpose(self.B) + SP[:,self.nD:]
        expected_SV     = tf.math.exp(SP[:,:self.nD]/2) @ tf.transpose(self.B) * SP[:,self.nD:]

        self.assertEqual(yspLG.shape, (2*(self.nD*2)+1, self.nD))
        self.assertEqual(yspSV.shape, (2*(self.nD*2)+1, self.nD))

        self.assertTrue(np.allclose(yspLG.numpy(), expected_LG.numpy()))
        self.assertTrue(np.allclose(yspSV.numpy(), expected_SV.numpy()))

    def test_ukf_predict_mean(self):
        wm, _, wi, _    = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        sp              = SigmaPoints(self.nD, self.Xprev, self.P, self.Lambda)
        mean            = UKF_Predict_mean(wm, wi, sp)
        expected_mean   = wm * sp[0, :] + tf.reduce_sum(wi * sp[1:, :], axis=0)
        
        self.assertEqual(mean.shape, (self.nD,))
        self.assertTrue(np.allclose(mean.numpy(), expected_mean.numpy()))
        self.assertTrue(isinstance(mean, tf.Tensor))


    def test_ukf_predict_cov(self):
        wm, wc, wi, _   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        sp              = SigmaPoints(self.nD, self.Xprev, self.P, self.Lambda)
        mean            = UKF_Predict_mean(wm, wi, sp)

        cov_result      = UKF_Predict_cov(self.nD, wc, wi, sp, mean, self.u)

        diffs           = sp - mean
        expected_cov    = wc * tf.tensordot(diffs[0, :], diffs[0, :], axes=0)
        for i in range(1, self.nD+1):
            expected_cov += wi * tf.tensordot(diffs[i, :], diffs[i, :], axes=0)
            expected_cov += wi * tf.tensordot(diffs[i+self.nD, :], diffs[i+self.nD, :], axes=0)

        self.assertEqual(cov_result.shape, (self.nD,self.nD))
        self.assertTrue(np.allclose(cov_result.numpy(), expected_cov.numpy()))
        self.assertTrue(isinstance(cov_result, tf.Tensor))


    def test_ukf_predict_crosscov(self):
        wm, wc, wi, _   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        sp              = SigmaPoints(self.nD, self.Xprev, self.P, self.Lambda)
        mean            = UKF_Predict_mean(wm, wi, sp)
        sp2             = SigmaPoints(self.nD, self.X, self.P, self.Lambda)
        mean2           = UKF_Predict_mean(wm, wi, sp)

        cov_result      = UKF_Predict_crosscov(self.nD, wc, wi, sp, mean, sp2, mean2, self.u)

        diffs           = sp - mean
        diffs2          = sp2 - mean2

        expected_cov    = wc * tf.tensordot(diffs[0, :], diffs2[0, :], axes=0)
        for i in range(1, self.nD+1):
            expected_cov += wi * tf.tensordot(diffs[i, :], diffs[i, :], axes=0)
            expected_cov += wi * tf.tensordot(diffs[i+self.nD, :], diffs2[i+self.nD, :], axes=0)

        self.assertEqual(cov_result.shape, (self.nD,self.nD))
        self.assertTrue(np.allclose(cov_result.numpy(), expected_cov.numpy()))
        self.assertTrue(isinstance(cov_result, tf.Tensor))


    def test_ukf_predict_add(self):
        wm, wc, wi, L   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        Xsp, Ysp, ypred, Wpred, Cpred = UKF_Predict_additive("LG", self.nD, self.X, self.P, wm, wc, wi, L, self.B, self.W, self.u) 

        self.assertEqual(Xsp, (self.nD*2+1,self.nD))
        self.assertEqual(Ysp, (self.nD*2+1,self.nD))
        self.assertEqual(ypred, (self.nD,))
        self.assertEqual(Wpred, (self.nD,self.nD))
        self.assertEqual(Cpred, (self.nD,self.nD))


    def test_ukf_predct_aug(self):
        wm, wc, wi, L   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        Xsp, Ysp, ypred, Wpred, Cpred = UKF_Predict_additive("SV", self.nD, self.X, self.P, wm, wc, wi, L, self.B, self.W, self.u) 

        self.assertEqual(Xsp, ((self.nD*2)*2+1,self.nD))
        self.assertEqual(Ysp, ((self.nD*2)*2+1,self.nD))
        self.assertEqual(ypred, (self.nD,))
        self.assertEqual(Wpred, (self.nD,self.nD))
        self.assertEqual(Cpred, (self.nD,self.nD))


    def test_ukf_gain(self):

        K               = UKF_Gain(self.P, self.W, self.u)
        expected_K      = self.P @ tf.linalg.inv(self.W + self.u)

        self.assertEqual(K.shape, (self.nD,self.nD))
        self.assertTrue(np.allclose(K.numpy(), expected_K.numpy()))
        self.assertTrue(isinstance(K, tf.Tensor))
        
    def test_ukf_filter(self):

        K               = UKF_Gain(self.P, self.W, self.u)
        x, P            = UKF_Filter(self.Xprev, self.P, self.W, self.X, self.Y, K)
        expected_x      = self.Xprev + tf.linalg.matvec(K, self.X - self.Y)
        expected_P      = self.P - K @ self.W @ tf.transpose(K)

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))
        
        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))

    def test_unscented_kalman_filter(self):

        _, Y            = SSM(self.nT, self.nD, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered      = UnscentedKalmanFilter(y=Y, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))

        _, Y2           = SSM(self.nT, self.nD, B=self.B, V=self.V, W=self.W)
        X_filtered2     = UnscentedKalmanFilter(y=Y2, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered2.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered2, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))



class TestPF(unittest.TestCase):

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.X          = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
        self.mu0        = tf.zeros((self.nD,), dtype=tf.float64)
        self.Sigma0     = (self.V @ self.V) @ tf.linalg.inv(tf.eye(self.nD, dtype=tf.float64) - self.A @ self.A) 
        
    def test_initiate_particles(self):

        particles       = initiate_particles(self.Np, self.nD, self.mu0, self.Sigma0)
        
        self.assertEqual(particles.shape, (self.Np, self.nD))
        self.assertTrue(isinstance(particles, tf.Variable))

    def test_log_importance(self):
        log_imp         = LogImportance(self.Xprev, self.mu0, self.Sigma0)
        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu0, covariance_matrix=self.Sigma0)
        log_imp2        = dist.log_prob(self.Xprev) + tf.math.log(2*pi_constant) * self.nD/2
        
        self.assertAlmostEqual(log_imp.numpy(), log_imp2.numpy(), places=5)

    def test_log_likelihood(self):

        log_like        = LogLikelihood(self.Y, self.mu0, self.W, self.u)
        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu0, covariance_matrix=self.W + self.u)
        log_like2       = dist.log_prob(self.Y) + tf.math.log(2*pi_constant) * self.nD/2
        
        self.assertAlmostEqual(log_like.numpy(), log_like2.numpy(), places=5)

    def test_log_target(self):
        log_tgt         = LogTarget(self.X, self.Xprev, self.Sigma0)
        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.Xprev, covariance_matrix=self.Sigma0)
        log_tgt2        = dist.log_prob(self.X) + tf.math.log(2*pi_constant) * self.nD/2
        
        self.assertAlmostEqual(log_tgt.numpy(), log_tgt2.numpy(), places=5)

    def test_draw_particles(self):
        particles       = initiate_particles(self.Np, self.nD, self.mu0, self.Sigma0)
        xn, Lp          = draw_particles(self.Np, self.nD, self.Y, particles, self.V, self.mu0, self.W, self.Sigma0, self.u)

        self.assertEqual(xn.shape, (self.Np, self.nD) )
        self.assertEqual(Lp.shape, (self.Np,))

    def test_compute_weights(self):
        w0              = tf.ones((self.Np,), dtype=tf.float64) / self.Np
        particles       = initiate_particles(self.Np, self.nD, self.mu0, self.Sigma0)
        _, Lp           = draw_particles(self.Np, self.nD, self.Y, particles, self.V, self.mu0, self.W, self.Sigma0, self.u)

        weights         = compute_weights(w0, Lp)
        expected_w      = tf.math.exp(tf.math.log(w0) + Lp)
        
        self.assertEqual(weights.shape, (self.Np,))
        self.assertTrue(np.allclose(weights.numpy(), expected_w.numpy()))

    def test_normalize_weights(self):
        w0              = tf.ones((self.Np,), dtype=tf.float64) / self.Np
        particles       = initiate_particles(self.Np, self.nD, self.mu0, self.Sigma0)
        _, Lp           = draw_particles(self.Np, self.nD, self.Y, particles, self.V, self.mu0, self.W, self.Sigma0, self.u)

        weights         = compute_weights(w0, Lp)
        norm_w          = normalize_weights(weights)

        self.assertAlmostEqual(tf.reduce_sum(norm_w).numpy(), 1.0, places=5)
        self.assertTrue(isinstance(norm_w, tf.Tensor))

    def test_compute_ess(self):
        weights         = tf.ones((self.Np,), dtype=tf.float64) / self.Np
        ess             = compute_ESS(weights)

        self.assertAlmostEqual(ess.numpy(), 1/(self.Np * (1/self.Np)**2) , places=5)
        self.assertTrue(isinstance(ess, tf.Tensor))

    def test_multinomial_resample(self):
        w0              = tf.ones((self.Np,), dtype=tf.float64) / self.Np
        particles       = initiate_particles(self.Np, self.nD, self.mu0, self.Sigma0)
        xbar, wbar      = multinomial_resample(self.Np, particles, w0)

        self.assertEqual(xbar.shape, (self.Np, self.nD))
        self.assertTrue(np.allclose(wbar.numpy(), tf.ones((self.Np,), dtype=tf.float64) / self.Np))
        self.assertAlmostEqual(tf.reduce_sum(wbar).numpy(), 1.0, places=5)
        self.assertTrue(isinstance(xbar, tf.Tensor))
        self.assertTrue(isinstance(wbar, tf.Tensor))

    def test_compute_posterior(self):
        w0              = tf.ones((self.Np,), dtype=tf.float64) / self.Np
        particles       = initiate_particles(self.Np, self.nD, self.mu0, self.Sigma0)
        posterior       = compute_posterior(w0, particles)
        posterior2      = tf.linalg.matvec(tf.transpose(particles), w0)
        
        self.assertEqual(posterior.shape, (self.nD,))
        self.assertTrue(np.allclose(posterior.numpy(), posterior2.numpy()))

    def test_particle_filter(self): 

        _, Y                                = SSM(self.nT, self.nD)
        X_filtered, ESS, Weights, xParts, xParts2    = ParticleFilter(y=Y, N=self.Np, resample=False)
        
        _, Y2                                = SSM(self.nT, self.nD, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered2, ESS2, Weights2, xPartsv2, xParts2v2    = ParticleFilter(y=Y2, model="SV", N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, resample=False)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertEqual(ESS.shape, (self.nT,))
        self.assertEqual(Weights.shape, (self.nT,self.Np))
        self.assertEqual(xParts.shape, (self.nT,self.Np,self.nD))
        self.assertEqual(xParts2.shape, (self.nT,self.Np,self.nD))

        self.assertEqual(X_filtered2.shape, (self.nT, self.nD))
        self.assertEqual(ESS2.shape, (self.nT,))
        self.assertEqual(Weights2.shape, (self.nT,self.Np))
        self.assertEqual(xPartsv2.shape, (self.nT,self.Np,self.nD))
        self.assertEqual(xParts2v2.shape, (self.nT,self.Np,self.nD))

        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(ESS, tf.Variable))
        self.assertTrue(isinstance(Weights, tf.Variable))
        self.assertTrue(isinstance(xParts, tf.Variable))
        self.assertTrue(isinstance(xParts2, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered2, tf.Variable))
        self.assertTrue(isinstance(ESS2, tf.Variable))
        self.assertTrue(isinstance(Weights2, tf.Variable))
        self.assertTrue(isinstance(xPartsv2, tf.Variable))
        self.assertTrue(isinstance(xParts2v2, tf.Variable))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xParts)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xParts2)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartsv2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xParts2v2)))


class TestEDH(unittest.TestCase): 

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
        self.mu0        = tf.zeros((self.nD,), dtype=tf.float64)
        self.Sigma0     = (self.V @ self.V) @ tf.linalg.inv(tf.eye(self.nD, dtype=tf.float64) - self.A @ self.A) 
        
        self.epsilon    = 0.1

        self.alpha      = 1.0
        self.kappa      = 3.0 * self.nD / 2.0
        self.beta       = 2.0
        self.Lambda     = (self.alpha ** 2) * self.kappa

        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        self.R, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) ) + tf.linalg.diag(1.0 + tf.range(self.nD, dtype=tf.float64))
        self.I          = tf.eye(self.nD, dtype=tf.float64)

    def test_eq10(self):

        result          = Li17eq10(1.0, self.I, self.P, self.R, self.u)
        result2         = - 1/2 * self.P @ tf.linalg.inv( self.P + self.R + self.u )

        self.assertEqual(result.shape, (self.nD, self.nD))
        self.assertTrue(np.allclose(result, result2, atol=1e-5))


    def test_eq11(self):
        e0              = tf.zeros((self.nD,), dtype=tf.float64)
        e1              = tf.ones((self.nD,), dtype=tf.float64)

        result          = Li17eq11(self.I, 1.0, self.A, self.I, self.P, self.R, self.Y, e0, e1, self.u)
        M               = (self.I + self.A) @ self.P @ tf.linalg.inv(self.R + self.u)
        result2         = tf.linalg.matvec(self.I + 2*self.A, tf.linalg.matvec(M, self.Y) + tf.linalg.matvec(self.A, e1))

        self.assertEqual(result.shape, (self.nD,))
        self.assertTrue(np.allclose(result, result2, atol=1e-5))

    def test_edh_linearize_ekf(self):

        particles       = initiate_particles(self.Np, self.nD, self.Xprev, self.P)
        results         = EDH_linearize_EKF(self.Np, self.nD, particles, self.Xprev, self.P, self.A, self.V, self.u)
        
        eta, eta0, m_pred, P_pred = results
        m2, P2          = EKF_Predict(self.Xprev, self.P, self.A, self.V)     

        self.assertEqual(eta.shape, (self.nD,))
        self.assertEqual(eta0.shape, (self.Np, self.nD))
        self.assertEqual(m_pred.shape, (self.nD,))
        self.assertEqual(P_pred.shape, (self.nD, self.nD))

        self.assertTrue(np.allclose(eta, m2, atol=1e-5))
        self.assertTrue(np.allclose(eta, m_pred, atol=1e-5))
        self.assertTrue(np.allclose(P_pred, P2, atol=1e-5))

    def test_edh_linearize_ukf(self):

        wm, wc, wi, L   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        particles       = initiate_particles(self.Np, self.nD, self.Xprev, self.P)
        results         = EDH_linearize_UKF(self.Np, self.nD, particles, self.Xprev, self.P, self.A, self.V, wm, wc, wi, L, self.u)
        
        eta, eta0, m_pred, P_pred = results
        xsp             = SigmaPoints(self.nD, self.Xprev, self.P, L) @ tf.transpose(self.A) 
        m2              = UKF_Predict_mean(wm, wi, xsp) 
        P2              = UKF_Predict_cov(self.nD, wc, wi, xsp, m2, self.u, Cov=self.V)  
        
        self.assertEqual(eta.shape, (self.nD,))
        self.assertEqual(eta0.shape, (self.Np, self.nD))
        self.assertEqual(m_pred.shape, (self.nD,))
        self.assertEqual(P_pred.shape, (self.nD, self.nD))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(eta0)))
        self.assertTrue(np.allclose(eta, m2, atol=1e-5))
        self.assertTrue(np.allclose(eta, m_pred, atol=1e-5))
        self.assertTrue(np.allclose(P_pred, P2, atol=1e-5))

    def test_edh_flow_dynamics(self):

        e00             = tf.zeros((self.nD,), dtype=tf.float64)
        e0              = tf.ones((self.Np, self.nD), dtype=tf.float64)
        e1              = tf.ones((self.nD,), dtype=tf.float64)
    
        res0, res       = EDH_flow_dynamics(self.Np, self.nD, 1.0, self.epsilon, self.I, e1, e0, self.P, self.I, self.R, e00, self.Y, self.u)

        Ci              = Li17eq10(1.0, self.I, self.P, self.R, self.u)
        bi              = Li17eq11(self.I, 1.0, Ci, self.I, self.P, self.R, self.Y, e00, e1, self.u)
        
        Res2            = self.epsilon * (tf.linalg.matvec(Ci, e1) + bi)
        Res02           = tf.Variable(tf.zeros((self.Np,self.nD), dtype=tf.float64))
        for i in range(self.Np): 
            Res02[i,:].assign(Res2)

        self.assertEqual(res0.shape, (self.Np, self.nD))
        self.assertEqual(res.shape, (self.nD,))

        self.assertTrue(np.allclose(res, Res2, atol=1e-5))
        self.assertTrue(np.allclose(res0, Res02, atol=1e-5))

    def test_edh_flow_lp(self):

        e0              = tf.zeros((self.Np, self.nD), dtype=tf.float64)
        Lp              = EDH_flow_lp(self.Np, e0, e0, e0, self.Y, self.V, self.mu0, self.W, self.u)

        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu0, covariance_matrix=self.W)
        log_like        = dist.log_prob(self.Y) + tf.math.log(2*pi_constant) * self.nD/2
        LogP            = tf.constant([log_like.numpy() for _ in range(self.Np)], dtype=tf.float64)
        
        self.assertEqual(Lp.shape, (self.Np,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Lp)))
        self.assertTrue(np.allclose(Lp, LogP, atol=1e-5))

    def test_edh(self):
        
        # _, Y                                    = SSM(self.nT, self.nD)
        # X_filtered, ESS, Weights, Jx, Jw        = EDH(y=Y, N=self.Np)
        # X_filtered2, ESS2, Weights2, Jx2, Jw2   = EDH(y=Y, N=self.Np, method="EKF")

        _, Y3                                   = SSM(self.nT, self.nD, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered3, ESS3, Weights3, Jx3, Jw3   = EDH(y=Y3, model="SV", N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered4, ESS4, Weights4, Jx4, Jw4   = EDH(y=Y3, model="SV", N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, method="EKF")

        # self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        # self.assertEqual(ESS.shape, (self.nT,))
        # self.assertEqual(Weights.shape, (self.nT,self.Np))
        # self.assertEqual(Jx.shape, (self.nT,self.nD, self.nD))
        # self.assertEqual(Jw.shape, (self.nT,self.nD, self.nD))
        
        # self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        # self.assertEqual(ESS2.shape, (self.nT,))
        # self.assertEqual(Weights2.shape, (self.nT,self.Np))
        # self.assertEqual(Jx2.shape, (self.nT,self.nD, self.nD))
        # self.assertEqual(Jw2.shape, (self.nT,self.nD, self.nD))

        self.assertEqual(X_filtered3.shape, (self.nT,self.nD))
        self.assertEqual(ESS3.shape, (self.nT,))
        self.assertEqual(Weights3.shape, (self.nT,self.Np))
        self.assertEqual(Jx3.shape, (self.nT,self.nD, self.nD))
        self.assertEqual(Jw3.shape, (self.nT,self.nD, self.nD))

        self.assertEqual(X_filtered4.shape, (self.nT,self.nD))
        self.assertEqual(ESS4.shape, (self.nT,))
        self.assertEqual(Weights4.shape, (self.nT,self.Np))
        self.assertEqual(Jx4.shape, (self.nT,self.nD, self.nD))
        self.assertEqual(Jw4.shape, (self.nT,self.nD, self.nD))

        # self.assertTrue(isinstance(X_filtered, tf.Variable))
        # self.assertTrue(isinstance(ESS, tf.Variable))
        # self.assertTrue(isinstance(Weights, tf.Variable))
        # self.assertTrue(isinstance(Jx, tf.Variable))
        # self.assertTrue(isinstance(Jw, tf.Variable))
        
        # self.assertTrue(isinstance(X_filtered2, tf.Variable))
        # self.assertTrue(isinstance(ESS2, tf.Variable))     
        # self.assertTrue(isinstance(Weights2, tf.Variable)) 
        # self.assertTrue(isinstance(Jx2, tf.Variable))
        # self.assertTrue(isinstance(Jw2, tf.Variable))

        self.assertTrue(isinstance(X_filtered3, tf.Variable))
        self.assertTrue(isinstance(ESS3, tf.Variable))
        self.assertTrue(isinstance(Weights3, tf.Variable))
        self.assertTrue(isinstance(Jx3, tf.Variable))
        self.assertTrue(isinstance(Jw3, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered4, tf.Variable))
        self.assertTrue(isinstance(ESS4, tf.Variable))     
        self.assertTrue(isinstance(Weights4, tf.Variable)) 
        self.assertTrue(isinstance(Jx4, tf.Variable))
        self.assertTrue(isinstance(Jw4, tf.Variable))

        # self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(Jx)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(Jw)))
        
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(Jx2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(Jw2)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jx3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jw3)))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jx4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jw4)))


class TestLEDH(unittest.TestCase): 

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
        self.mu0        = tf.zeros((self.nD,), dtype=tf.float64)
        self.Sigma0     = (self.V @ self.V) @ tf.linalg.inv(tf.eye(self.nD, dtype=tf.float64) - self.A @ self.A) 
        
        self.epsilon    = 0.1

        self.alpha      = 1.0
        self.kappa      = 3.0 * self.nD / 2.0
        self.beta       = 2.0
        self.Lambda     = (self.alpha ** 2) * self.kappa

        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        self.R          = tf.linalg.diag(tf.range(self.nD, dtype=tf.float64) + 1.0)
        self.I          = tf.eye(self.nD, dtype=tf.float64)
        self.I1         = tf.ones((self.nD,), dtype=tf.float64)

    def test_ledh_linearize_ekf(self):
        x0              = tf.zeros((self.Np,self.nD), dtype=tf.float64)
        Iarr            = self.I.numpy()
        P0              = tf.constant([Iarr for _ in range(self.Np)], dtype=tf.float64)
        
        results         = LEDH_linearize_EKF(self.Np, self.nD, "SV", self.I1, x0, P0, self.A, self.B, self.V, self.W, self.mu0, self.u)
        e, e0, mpred, Ppred, ypred, Hx, Hw, R, el = results 
        
        results2        = LEDH_linearize_EKF(self.Np, self.nD, "LG", self.I1, x0, P0, self.A, self.B, self.V, self.W, self.mu0, self.u)
        e2, e02, mpred2, Ppred2, ypred2, Hx2, Hw2, R2, el2 = results2 

        Parr            = (self.A @ self.A + self.V).numpy()
        P2              = tf.constant([Parr for _ in range(self.Np)], dtype=tf.float64)
        
        self.assertEqual(e0.shape, (self.Np,self.nD))
        self.assertEqual(ypred.shape, (self.Np,self.nD))
        self.assertEqual(Hx.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(Hw.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(R.shape, (self.Np,self.nD,self.nD)) 
        self.assertEqual(el.shape, (self.Np,self.nD)) 
        
        self.assertEqual(e02.shape, (self.Np,self.nD))
        self.assertEqual(ypred2.shape, (self.Np,self.nD))
        self.assertEqual(Hx2.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(Hw2.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(R2.shape, (self.Np,self.nD,self.nD)) 
        self.assertEqual(el2.shape, (self.Np,self.nD)) 

        self.assertTrue( np.allclose(e, tf.zeros((self.Np, self.nD))) )
        self.assertTrue( np.allclose(mpred, tf.zeros((self.Np, self.nD))) )
        self.assertTrue( np.allclose(Ppred, P2) )

        self.assertTrue( np.allclose(e2, tf.zeros((self.Np, self.nD))) )
        self.assertTrue( np.allclose(mpred2, tf.zeros((self.Np, self.nD))) )
        self.assertTrue( np.allclose(Ppred2, P2) )


    def test_ledh_linearize_ukf(self):
        
        wm, wc, wi, L   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        x0              = tf.zeros((self.Np,self.nD), dtype=tf.float64)
        P0              = tf.constant([self.P.numpy() for _ in range(self.Np)], dtype=tf.float64)

        results         = LEDH_linearize_UKF(self.Np, self.nD, "SV", self.I1, x0, P0, self.A, self.B, self.V, self.W, wm, wc, wi, L, self.u)
        e, e0, mpred, Ppred, y, H, Hw, Cy, err, xSP, ySP = results
        
        results2        = LEDH_linearize_UKF(self.Np, self.nD, "LG", self.I1, x0, P0, self.A, self.B, self.V, self.W, wm, wc, wi, L, self.u)
        eLG, e0LG, mpredLG, PpredLG, yLG, HLG, HwLG, CyLG, errLG, xSPLG, ySPLG = results2

        e2              = tf.Variable(tf.zeros((self.Np,self.nD), dtype=tf.float64))
        mpred2          = tf.Variable(tf.zeros((self.Np,self.nD), dtype=tf.float64))
        Ppred2          = tf.Variable(tf.zeros((self.Np,self.nD,self.nD), dtype=tf.float64))
        for i in range(self.Np): 
            
            Xprev_sp    = SigmaPoints(self.nD, x0[i,:], P0[i,:,:], L)
            X_sp        = Xprev_sp @ tf.transpose(self.A) 
            mp          = UKF_Predict_mean(wm, wi, X_sp) 
            Pp          = UKF_Predict_cov(self.nD, wc, wi, X_sp, mp, self.u, Cov=self.V)  
        
            e2[i,:].assign(mp)    
            mpred2[i,:].assign(mp)
            Ppred2[i,:,:].assign(Pp)
            
        self.assertEqual(e.shape, (self.Np,self.nD))
        self.assertEqual(e0.shape, (self.Np,self.nD))
        self.assertEqual(mpred.shape, (self.Np,self.nD))
        self.assertEqual(Ppred.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(y.shape, (self.Np,self.nD))
        self.assertEqual(H.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(Hw.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(Cy.shape, (self.Np,self.nD,self.nD)) 
        self.assertEqual(err.shape, (self.Np,self.nD)) 
        self.assertEqual(xSP.shape, (self.Np,2*(2*self.nD)+1,2*self.nD))
        self.assertEqual(ySP.shape, (self.Np,2*(2*self.nD)+1,self.nD)) 

        self.assertEqual(eLG.shape, (self.Np,self.nD))
        self.assertEqual(e0LG.shape, (self.Np,self.nD))
        self.assertEqual(mpredLG.shape, (self.Np,self.nD))
        self.assertEqual(PpredLG.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(yLG.shape, (self.Np,self.nD))
        self.assertEqual(HLG.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(HwLG.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(CyLG.shape, (self.Np,self.nD,self.nD)) 
        self.assertEqual(errLG.shape, (self.Np,self.nD)) 
        self.assertEqual(xSPLG.shape, (self.Np,2*(2*self.nD)+1,2*self.nD))
        self.assertEqual(ySPLG.shape, (self.Np,2*(2*self.nD)+1,self.nD)) 

        self.assertTrue( np.allclose(e, e2) )
        self.assertTrue( np.allclose(mpred, mpred2) )
        self.assertTrue( np.allclose(Ppred, Ppred2) )

        self.assertTrue( np.allclose(eLG, e2) )
        self.assertTrue( np.allclose(mpredLG, mpred2) )
        self.assertTrue( np.allclose(PpredLG, Ppred2) )


    def test_ledh_flow_dynamics(self):

        e00             = tf.zeros((self.Np, self.nD), dtype=tf.float64)
        e0              = tf.ones((self.Np, self.nD), dtype=tf.float64)
        e1              = tf.ones((self.Np, self.nD), dtype=tf.float64)
        
        P0              = tf.constant([self.P.numpy() for _ in range(self.Np)], dtype=tf.float64)
        Iarr            = tf.constant([self.I.numpy() for _ in range(self.Np)], dtype=tf.float64)
        R0              = tf.constant([self.R.numpy() for _ in range(self.Np)], dtype=tf.float64)
        
        res0, res, resp = LEDH_flow_dynamics(self.Np, self.nD, 1.0, self.epsilon, self.I, e1, e0, P0, Iarr, R0, e00, self.Y, self.u)

        Ci              = Li17eq10(1.0, self.I, self.P, self.R, self.u)
        bi              = Li17eq11(self.I, 1.0, Ci, self.I, self.P, self.R, self.Y, e00[0,:], e1[0,:], self.u)
        
        Resi            = self.epsilon * (tf.linalg.matvec(Ci, e1[0,:]) + bi)
        Res2            = tf.Variable(tf.zeros((self.Np,self.nD), dtype=tf.float64))
        Res02           = tf.Variable(tf.zeros((self.Np,self.nD), dtype=tf.float64))
        for i in range(self.Np): 
            Res2[i,:].assign(Resi)
            Res02[i,:].assign(Resi)

        self.assertEqual(res0.shape, (self.Np, self.nD))
        self.assertEqual(res.shape, (self.Np, self.nD))
        self.assertEqual(resp.shape, (self.Np,))

        self.assertTrue(np.allclose(res, Res2, atol=1e-5))
        self.assertTrue(np.allclose(res0, Res02, atol=1e-5))
        
    def test_ledh_flow_lp(self):
        
        t0              = tf.ones((self.Np,), dtype=tf.float64) 
        e0              = tf.zeros((self.Np, self.nD), dtype=tf.float64)
        Sx              = tf.constant([self.V.numpy() for _ in range(self.Np)], dtype=tf.float64)
        Sw              = tf.constant([self.W.numpy() for _ in range(self.Np)], dtype=tf.float64)
        
        Lp              = LEDH_flow_lp(self.Np, e0, t0, e0, e0, self.Y, Sx, self.mu0, Sw, self.u)

        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu0, covariance_matrix=self.W)
        log_like        = dist.log_prob(self.Y) + tf.math.log(2*pi_constant) * self.nD/2 
        LogP            = tf.constant([log_like.numpy() for _ in range(self.Np)], dtype=tf.float64)
        
        self.assertEqual(Lp.shape, (self.Np,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Lp)))
        self.assertTrue(np.allclose(Lp, LogP, atol=1e-5))
        
    def test_ledh(self):
        
        # _, Y                                        = SSM(self.nT, self.nD)
        # X_filtered, ESS, Weights, JxC, JwC          = LEDH(y=Y, N=self.Np)
        # X_filtered2, ESS2, Weights2, JxC2, JwC2     = LEDH(y=Y, N=self.Np, method="EKF")

        _, Y2                                       = SSM(self.nT, self.nD, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered3, ESS3, Weights3, JxC3, JwC3     = LEDH(y=Y2, N=self.Np, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered4, ESS4, Weights4, JxC4, JwC4     = LEDH(y=Y2, N=self.Np, model="SV", A=self.A, B=self.B, V=self.V, W=self.W, method="EKF")

        # self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        # self.assertEqual(ESS.shape, (self.nT,))
        # self.assertEqual(Weights.shape, (self.nT,self.Np))        
        # self.assertEqual(JxC.shape, (self.nT,self.Np, self.nD, self.nD))
        # self.assertEqual(JwC.shape, (self.nT,self.Np, self.nD, self.nD))
        
        # self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        # self.assertEqual(ESS2.shape, (self.nT,))
        # self.assertEqual(Weights2.shape, (self.nT,self.Np))        
        # self.assertEqual(JxC2.shape, (self.nT,self.Np, self.nD, self.nD))
        # self.assertEqual(JwC2.shape, (self.nT,self.Np, self.nD, self.nD))


        self.assertEqual(X_filtered3.shape, (self.nT,self.nD))
        self.assertEqual(ESS3.shape, (self.nT,))
        self.assertEqual(Weights3.shape, (self.nT,self.Np))        
        self.assertEqual(JxC3.shape, (self.nT,self.Np, self.nD, self.nD))
        self.assertEqual(JwC3.shape, (self.nT,self.Np, self.nD, self.nD))
        
        self.assertEqual(X_filtered4.shape, (self.nT,self.nD))
        self.assertEqual(ESS4.shape, (self.nT,))
        self.assertEqual(Weights4.shape, (self.nT,self.Np))        
        self.assertEqual(JxC4.shape, (self.nT,self.Np, self.nD, self.nD))
        self.assertEqual(JwC4.shape, (self.nT,self.Np, self.nD, self.nD))

        # self.assertTrue(isinstance(X_filtered, tf.Variable))
        # self.assertTrue(isinstance(ESS, tf.Variable))
        # self.assertTrue(isinstance(Weights, tf.Variable))
        # self.assertTrue(isinstance(JxC, tf.Variable))
        # self.assertTrue(isinstance(JwC, tf.Variable))
        
        # self.assertTrue(isinstance(X_filtered2, tf.Variable))
        # self.assertTrue(isinstance(ESS2, tf.Variable))    
        # self.assertTrue(isinstance(Weights2, tf.Variable))
        # self.assertTrue(isinstance(JxC2, tf.Variable))
        # self.assertTrue(isinstance(JwC2, tf.Variable))    

        
        self.assertTrue(isinstance(X_filtered3, tf.Variable))
        self.assertTrue(isinstance(ESS3, tf.Variable))
        self.assertTrue(isinstance(Weights3, tf.Variable))
        self.assertTrue(isinstance(JxC3, tf.Variable))
        self.assertTrue(isinstance(JwC3, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered4, tf.Variable))
        self.assertTrue(isinstance(ESS4, tf.Variable))    
        self.assertTrue(isinstance(Weights4, tf.Variable))
        self.assertTrue(isinstance(JxC4, tf.Variable))
        self.assertTrue(isinstance(JwC4, tf.Variable))    

        # self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(Weights)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC)))
        
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(Weights2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC2)))
        # self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC2)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Weights3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC3)))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Weights4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC4)))





class TestKernelPFF(unittest.TestCase): 

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.99, 0.99, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
        self.mu0        = tf.zeros((self.nD,), dtype=tf.float64)
        self.Sigma0     = (self.V @ self.V) @ tf.linalg.inv(tf.eye(self.nD, dtype=tf.float64) - self.A @ self.A) 
        
        self.epsilon    = 0.1
        self.Nl         = 30

        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        self.R          = tf.linalg.diag(tf.range(self.nD, dtype=tf.float64) + 1.0)
        self.I          = tf.eye(self.nD, dtype=tf.float64)
        self.I1         = tf.ones((self.nD,), dtype=tf.float64)
        
    def test_eq13(self):
        e1              = tf.ones((self.nD,), dtype=tf.float64)

        A               = Hu21eq13("SV", self.nD, e1, self.mu0, self.I, self.P, self.I, self.u)
        A2              = tf.linalg.matvec( tf.linalg.inv( self.P @ tf.transpose(self.P) + self.u), e1)
        
        A3              = Hu21eq13("LG", self.nD, e1, self.mu0, self.I, self.P, self.I, self.u)
        A4              = tf.linalg.matvec( tf.transpose(self.I) @ tf.linalg.inv(self.P), e1 - self.mu0)          
        
        self.assertTrue(np.allclose(A,A2))
        self.assertTrue(np.allclose(A3,A4))
        
    def test_eq15(self):
        e1              = tf.ones((self.nD,), dtype=tf.float64)
        A               = Hu21eq15(e1, self.mu0, self.P, self.u)
        A2              = tf.linalg.matvec( tf.linalg.inv(self.P + self.u), e1 )
        
        self.assertTrue(isinstance(A, tf.Tensor))
        self.assertTrue( np.allclose(A,A2) )
        
    def test_kpff_lp(self):

        e0              = tf.zeros((self.Np,self.nD), dtype=tf.float64)
        lp, jxc, jwc    = KPFF_LP(self.Np, self.nD, self.nD, "SV", self.I1, e0, self.mu0, self.mu0, self.B, self.W, self.mu0, self.Sigma0, self.u, self.u)
        lp2, jxc2, jwc2 = KPFF_LP(self.Np, self.nD, self.nD, "LG", self.I1, e0, self.mu0, self.mu0, self.B, self.W, self.mu0, self.Sigma0, self.u, self.u)

        self.assertEqual(lp.shape, (self.Np,self.nD))
        self.assertTrue(isinstance(lp, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(lp)))
        
        self.assertEqual(lp2.shape, (self.Np,self.nD))
        self.assertTrue(isinstance(lp2, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(lp2)))

        self.assertEqual(jxc.shape, (self.Np, self.nD, self.nD))
        self.assertTrue(isinstance(jxc, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jxc)))
        
        self.assertEqual(jxc2.shape, (self.Np, self.nD, self.nD))
        self.assertTrue(isinstance(jxc2, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jxc2)))

        self.assertEqual(jwc.shape, (self.Np, self.nD, self.nD))
        self.assertTrue(isinstance(jwc, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jwc)))
        
        self.assertEqual(jwc2.shape, (self.Np, self.nD, self.nD))
        self.assertTrue(isinstance(jwc2, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jwc2)))

    def test_kpff_rkhs(self):
        e0              = tf.zeros((self.Np, self.nD), dtype=tf.float64)
        e1              = tf.ones((self.Np, self.nD), dtype=tf.float64)
        
        II              = KPFF_RKHS(self.Np, self.nD, e0, e1, self.I)
        
        k, kc           = SE_Cov_div(self.Np, e0[:,0], 1.0)
        ks              = tf.reduce_sum( (k + kc*k) / self.Np, axis=1)
        II2             = tf.constant([ks.numpy() for _ in range(self.nD)], dtype=tf.float64, shape=(self.Np,self.nD))
        
        self.assertTrue(isinstance(II, tf.Variable))
        self.assertTrue( np.allclose(II,II2) )
        
    def test_kpff_flow(self):
        e1              = tf.ones((self.Np, self.nD), dtype=tf.float64)
        F               = KPFF_flow(self.Np, self.nD, self.epsilon, e1, self.I)
        s               = tf.linalg.matvec(self.I, e1[0,:]) 
        F2              = tf.constant([(self.epsilon * s).numpy() for _ in range(self.Np)], dtype=tf.float64, shape=(self.Np, self.nD))

        self.assertTrue(isinstance(F, tf.Variable))
        self.assertTrue( np.allclose(F, F2))

    def test_kernel_pff(self):
        
        _, Y                                        = SSM(self.nT, self.nD, model="SV", A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered, JxC, JwC, xPart, xPartp         = KernelPFF(y=Y, Nx=self.nD, model="SV", N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered2, JxC2, JwC2, xPart2, xPartp2    = KernelPFF(y=Y, Nx=self.nD, model="SV", N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, method="scalar")

        _, Y2                                       = SSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered3, JxC3, JwC3, xPart3, xPartp3    = KernelPFF(y=Y2, Nx=self.nD, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered4, JxC4, JwC4, xPart4, xPartp4    = KernelPFF(y=Y2, Nx=self.nD, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, method="scalar")

        self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        self.assertEqual(JxC.shape, (self.nT, self.Nl, self.Np, self.nD, self.nD))
        self.assertEqual(JwC.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(xPart.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp.shape, (self.nT,self.Np, self.nD))
        
        self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        self.assertEqual(JxC2.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(JwC2.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(xPart2.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp2.shape, (self.nT,self.Np, self.nD))
       
        self.assertEqual(X_filtered3.shape, (self.nT,self.nD))
        self.assertEqual(JxC3.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(JwC3.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(xPart3.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp3.shape, (self.nT,self.Np, self.nD))
       
        self.assertEqual(X_filtered4.shape, (self.nT,self.nD))
        self.assertEqual(JxC4.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(JwC4.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(xPart4.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp4.shape, (self.nT,self.Np, self.nD))
       
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(JxC, tf.Variable))
        self.assertTrue(isinstance(JwC, tf.Variable))
        self.assertTrue(isinstance(xPart, tf.Variable))
        self.assertTrue(isinstance(xPartp, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered2, tf.Variable)) 
        self.assertTrue(isinstance(JxC2, tf.Variable))
        self.assertTrue(isinstance(JwC2, tf.Variable))
        self.assertTrue(isinstance(xPart2, tf.Variable))
        self.assertTrue(isinstance(xPartp2, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered3, tf.Variable)) 
        self.assertTrue(isinstance(JxC3, tf.Variable))
        self.assertTrue(isinstance(JwC3, tf.Variable))
        self.assertTrue(isinstance(xPart3, tf.Variable))
        self.assertTrue(isinstance(xPartp3, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered4, tf.Variable)) 
        self.assertTrue(isinstance(JxC4, tf.Variable))
        self.assertTrue(isinstance(JwC4, tf.Variable))
        self.assertTrue(isinstance(xPart4, tf.Variable))
        self.assertTrue(isinstance(xPartp4, tf.Variable))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp)))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp2)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp3)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp4)))

    def test_kernel_pff_sparse(self):
        
        nS                                          = self.nD - 1
        
        _, Y                                        = SSM(self.nT, self.nD, n_sparse=nS, model="SV", V=self.V)
        X_filtered, JxC, JwC, xPart, xPartp         = KernelPFF(y=Y, Nx=self.nD, N=self.Np, V=self.V)
        X_filtered2, JxC2, JwC2, xPart2, xPartp2    = KernelPFF(y=Y, Nx=self.nD, N=self.Np, V=self.V, method="scalar")

        _, Y2                                       = SSM(self.nT, self.nD, n_sparse=nS, A=self.A, V=self.V)
        X_filtered3, JxC3, JwC3, xPart3, xPartp3    = KernelPFF(y=Y2, Nx=self.nD, N=self.Np, A=self.A, V=self.V)
        X_filtered4, JxC4, JwC4, xPart4, xPartp4    = KernelPFF(y=Y2, Nx=self.nD, N=self.Np, A=self.A, V=self.V, method="scalar")

        self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        self.assertEqual(JxC.shape, (self.nT, self.Nl, self.Np, nS, self.nD))
        self.assertEqual(JwC.shape, (self.nT,self.Nl,self.Np, nS, nS))
        self.assertEqual(xPart.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp.shape, (self.nT,self.Np, self.nD))
        
        self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        self.assertEqual(JxC2.shape, (self.nT,self.Nl,self.Np, nS, self.nD))
        self.assertEqual(JwC2.shape, (self.nT,self.Nl,self.Np, nS, nS))
        self.assertEqual(xPart2.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp2.shape, (self.nT,self.Np, self.nD))
       
        self.assertEqual(X_filtered3.shape, (self.nT,self.nD))
        self.assertEqual(JxC3.shape, (self.nT,self.Nl,self.Np, nS, self.nD))
        self.assertEqual(JwC3.shape, (self.nT,self.Nl,self.Np, nS, nS))
        self.assertEqual(xPart3.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp3.shape, (self.nT,self.Np, self.nD))
       
        self.assertEqual(X_filtered4.shape, (self.nT,self.nD))
        self.assertEqual(JxC4.shape, (self.nT,self.Nl,self.Np, nS, self.nD))
        self.assertEqual(JwC4.shape, (self.nT,self.Nl,self.Np, nS, nS))
        self.assertEqual(xPart4.shape, (self.nT,self.Np, self.nD))
        self.assertEqual(xPartp4.shape, (self.nT,self.Np, self.nD))
       
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(JxC, tf.Variable))
        self.assertTrue(isinstance(JwC, tf.Variable))
        self.assertTrue(isinstance(xPart, tf.Variable))
        self.assertTrue(isinstance(xPartp, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered2, tf.Variable)) 
        self.assertTrue(isinstance(JxC2, tf.Variable))
        self.assertTrue(isinstance(JwC2, tf.Variable))
        self.assertTrue(isinstance(xPart2, tf.Variable))
        self.assertTrue(isinstance(xPartp2, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered3, tf.Variable)) 
        self.assertTrue(isinstance(JxC3, tf.Variable))
        self.assertTrue(isinstance(JwC3, tf.Variable))
        self.assertTrue(isinstance(xPart3, tf.Variable))
        self.assertTrue(isinstance(xPartp3, tf.Variable))
                
        self.assertTrue(isinstance(X_filtered4, tf.Variable)) 
        self.assertTrue(isinstance(JxC4, tf.Variable))
        self.assertTrue(isinstance(JwC4, tf.Variable))
        self.assertTrue(isinstance(xPart4, tf.Variable))
        self.assertTrue(isinstance(xPartp4, tf.Variable))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp)))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp2)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart3)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp3)))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPart4)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xPartp4)))

        
if __name__ == '__main__':
    unittest.main()
