#################
# Set directory #
#################

path                = "C:/Users/anastasia/MyProjects/Codebase/ParticleFilteringJPM"
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
import tensorflow_probability as tfp
tfd = tfp.distributions
import math
pi_constant = tf.constant(math.pi, dtype=tf.float64)

import unittest
from functions import SE_kernel, SE_kernel_divC, SE_Cov_div
from functions import norm_rvs, LGSSM, SV_transform, SVSSM

from functions import KF_Predict, KF_Gain, KF_Filter, KalmanFilter
from functions import EKF_Predict, EKF_Jacobi, EKF_Gain, EKF_Filter, ExtendedKalmanFilter
from functions import SigmaWeights, SigmaPoints, UKF_Predict_mean, UKF_Predict_cov, UKF_Predict_crosscov, UKF_Gain, UKF_Filter, UnscentedKalmanFilter

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
        self.nT         = 10
        self.X          = tf.random.normal((self.nD,), dtype=tf.float64)

        self.mean       = tf.zeros((self.nD,), dtype=tf.float64)
        self.Sigma      = tf.eye(self.nD, dtype=tf.float64)

        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))


    def test_norm_rvs(self):
        sample          = norm_rvs(self.nD, self.mean, self.Sigma)

        self.assertEqual(sample.shape, (self.nD,))
        self.assertTrue(isinstance(sample, tf.Tensor))

    def test_lgssm(self):
        X, Y            = LGSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X.shape, (self.nT, self.nD))
        self.assertEqual(Y.shape, (self.nT, self.nD))

        self.assertTrue(isinstance(X, tf.Variable))
        self.assertTrue(isinstance(Y, tf.Variable))

    def test_sv_transform(self):
        sample          = SV_transform(self.nD, self.mean, self.B, self.X, self.W)

        self.assertEqual(sample.shape, (self.nD,))
        self.assertTrue(isinstance(sample, tf.Tensor))
        
    def test_svssm(self):

        X, Y            = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X.shape, (self.nT, self.nD))
        self.assertEqual(Y.shape, (self.nT, self.nD))

        self.assertTrue(isinstance(X, tf.Variable))
        self.assertTrue(isinstance(Y, tf.Variable))


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
        
        _, Y            = LGSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered      = KalmanFilter(y=Y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))



class TestEKF(unittest.TestCase):

    def setUp(self):

        self.nD         = 3
        self.nT         = 10
        self.Xprev          = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
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

    def test_ekf_jacobi(self):

        y_pred          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = EKF_Jacobi(self.Xprev, y_pred, self.B)
        
        expected_Jx     = tf.linalg.diag(y_pred / 2)
        expected_Jw     = tf.linalg.diag(tf.linalg.matvec(self.B, tf.math.exp(self.Xprev / 2)))

        self.assertEqual(Jx.shape, (self.nD, self.nD))
        self.assertEqual(Jw.shape, (self.nD, self.nD))

        self.assertTrue(np.allclose(Jx.numpy(), expected_Jx.numpy()))
        self.assertTrue(np.allclose(Jw.numpy(), expected_Jw.numpy()))

    def test_ekf_gain(self):

        ypred           = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = EKF_Jacobi(self.Xprev, ypred, self.B)
        K               = EKF_Gain(self.P, Jx, Jw, self.W, self.u)
        
        Mx              = self.P @ tf.transpose(Jx)
        J               = Jx @ Mx + Jw @ self.W @ tf.transpose(Jw)
        Minv            = tf.linalg.inv(J + self.u)
        expected_K      = Mx @ Minv

        self.assertTrue(np.allclose(K.numpy(), expected_K.numpy()))
        self.assertTrue(isinstance(K, tf.Tensor))
        
    def test_ekf_filter(self):

        y_pred          = tf.constant(tf.range(self.nD, dtype=tf.float64))
        Jx, Jw          = EKF_Jacobi(self.Xprev, y_pred, self.B)
        K               = EKF_Gain(self.P, Jx, Jw, self.W, self.u)
        x, P            = EKF_Filter(self.Xprev, self.P, self.Y, y_pred, Jx, K)

        expected_x      = self.Xprev + tf.linalg.matvec(K, self.Y - y_pred)
        expected_P      = self.P - self.P @ tf.transpose(Jx) @ tf.transpose(K)

        self.assertTrue(np.allclose(x.numpy(), expected_x.numpy()))
        self.assertTrue(np.allclose(P.numpy(), expected_P.numpy()))
        
        self.assertTrue(isinstance(x, tf.Tensor))
        self.assertTrue(isinstance(P, tf.Tensor))

    def test_extended_kalman_filter(self):

        _, Y            = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered      = ExtendedKalmanFilter(y=Y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))



class TestUKF(unittest.TestCase):

    def setUp(self):
        
        self.nD         = 3
        self.nT         = 10

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.X          = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

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
        for i in range(1, self.nD):
            expected_cov += wi * tf.tensordot(diffs[i, :], diffs[i, :], axes=0)

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
        for i in range(1, self.nD):
            expected_cov += wi * tf.tensordot(diffs[i, :], diffs2[i, :], axes=0)

        self.assertEqual(cov_result.shape, (self.nD,self.nD))
        self.assertTrue(np.allclose(cov_result.numpy(), expected_cov.numpy()))
        self.assertTrue(isinstance(cov_result, tf.Tensor))

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

        _, Y            = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered      = UnscentedKalmanFilter(y=Y, A=self.A, B=self.B, V=self.V, W=self.W)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))




class TestPF(unittest.TestCase):

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.X          = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
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
        log_like        = LogLikelihood(self.Xprev, self.Y, self.mu0, self.W, self.u)
        xe              = tf.math.exp(self.Xprev/2)
        Ci              = tf.linalg.diag(xe) @ self.W @ tf.linalg.diag(xe) + self.u
        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu0, covariance_matrix=Ci)
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

        _, Y                                = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered, ESS, Weights, xParts    = ParticleFilter(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, resample=False)

        self.assertEqual(X_filtered.shape, (self.nT, self.nD))
        self.assertEqual(ESS.shape, (self.nT,))
        self.assertEqual(Weights.shape, (self.nT,self.Np))
        self.assertEqual(xParts.shape, (self.nT,self.Np,self.nD))

        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(ESS, tf.Variable))
        self.assertTrue(isinstance(Weights, tf.Variable))
        self.assertTrue(isinstance(xParts, tf.Variable))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(xParts)))



class TestEDH(unittest.TestCase): 

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
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
        
        _, Y                                    = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered, ESS, Weights, Jx, Jw        = EDH(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, stepsize=self.epsilon)
        X_filtered2, ESS2, Weights2, Jx2, Jw2   = EDH(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, stepsize=self.epsilon, method="EKF")

        self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        self.assertEqual(ESS.shape, (self.nT,))
        self.assertEqual(Weights.shape, (self.nT,self.Np))
        self.assertEqual(Jx.shape, (self.nT,self.nD, self.nD))
        self.assertEqual(Jw.shape, (self.nT,self.nD, self.nD))
        
        
        self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        self.assertEqual(ESS2.shape, (self.nT,))
        self.assertEqual(Weights2.shape, (self.nT,self.Np))
        self.assertEqual(Jx2.shape, (self.nT,self.nD, self.nD))
        self.assertEqual(Jw2.shape, (self.nT,self.nD, self.nD))

        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(ESS, tf.Variable))
        self.assertTrue(isinstance(Weights, tf.Variable))
        self.assertTrue(isinstance(Jx, tf.Variable))
        self.assertTrue(isinstance(Jw, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered2, tf.Variable))
        self.assertTrue(isinstance(ESS2, tf.Variable))     
        self.assertTrue(isinstance(Weights2, tf.Variable)) 
        self.assertTrue(isinstance(Jx2, tf.Variable))
        self.assertTrue(isinstance(Jw2, tf.Variable))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jx)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jw)))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jx2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Jw2)))






class TestLEDH(unittest.TestCase): 

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
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

    def test_ledh_linearize_ekf(self):
        x0              = tf.zeros((self.Np,self.nD), dtype=tf.float64)
        Iarr            = self.I.numpy()
        P0              = tf.constant([Iarr for _ in range(self.Np)], dtype=tf.float64)
        
        results         = LEDH_linearize_EKF(self.Np, self.nD, x0, P0, self.A, self.B, self.V, self.W, self.mu0, self.u)
        e, e0, mpred, Ppred, ypred, Hx, Hw, R, el = results 
        
        Parr            = (self.A @ self.A + self.V).numpy()
        P2              = tf.constant([Parr for _ in range(self.Np)], dtype=tf.float64)
        
        self.assertEqual(e0.shape, (self.Np,self.nD))
        self.assertEqual(ypred.shape, (self.Np,self.nD))
        self.assertEqual(Hx.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(Hw.shape, (self.Np,self.nD,self.nD))
        self.assertEqual(R.shape, (self.Np,self.nD,self.nD)) 
        self.assertEqual(el.shape, (self.Np,self.nD)) 
        
        self.assertTrue( np.allclose(e, tf.zeros((self.Np, self.nD))) )
        self.assertTrue( np.allclose(mpred, tf.zeros((self.Np, self.nD))) )
        self.assertTrue( np.allclose(Ppred, P2) )



    def test_ledh_linearize_ukf(self):
        
        wm, wc, wi, L   = SigmaWeights(self.nD, self.alpha, self.kappa, self.beta)
        x0              = tf.zeros((self.Np,self.nD), dtype=tf.float64)
        P0              = tf.constant([self.P.numpy() for _ in range(self.Np)], dtype=tf.float64)
        
        results         = LEDH_linearize_UKF(self.Np, self.nD, x0, P0, self.A, self.B, self.V, self.W, wm, wc, wi, L, self.u)
        e, e0, mpred, Ppred, y, H, Hw, Cy, err, xSP, ySP = results
        
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

        self.assertTrue( np.allclose(e, e2) )
        self.assertTrue( np.allclose(mpred, mpred2) )
        self.assertTrue( np.allclose(Ppred, Ppred2) )


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
        
        Lp              = LEDH_flow_lp(self.Np, e0, t0, e0, e0, self.Y, Sx, self.mu0, self.W, self.u)

        dist            = tfp.distributions.MultivariateNormalFullCovariance(loc=self.mu0, covariance_matrix=self.W)
        log_like        = dist.log_prob(self.Y) + tf.math.log(2*pi_constant) * self.nD/2 
        LogP            = tf.constant([log_like.numpy() for _ in range(self.Np)], dtype=tf.float64)
        
        self.assertEqual(Lp.shape, (self.Np,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Lp)))
        self.assertTrue(np.allclose(Lp, LogP, atol=1e-5))
        
    def test_ledh(self):
        
        _, Y                                        = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered, ESS, Weights, JxC, JwC          = LEDH(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, stepsize=self.epsilon)
        X_filtered2, ESS2, Weights2, JxC2, JwC2     = LEDH(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, stepsize=self.epsilon, method="EKF")

        self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        self.assertEqual(ESS.shape, (self.nT,))
        self.assertEqual(Weights.shape, (self.nT,self.Np))        
        self.assertEqual(JxC.shape, (self.nT,self.Np, self.nD, self.nD))
        self.assertEqual(JwC.shape, (self.nT,self.Np, self.nD, self.nD))
        
        self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        self.assertEqual(ESS2.shape, (self.nT,))
        self.assertEqual(Weights2.shape, (self.nT,self.Np))        
        self.assertEqual(JxC2.shape, (self.nT,self.Np, self.nD, self.nD))
        self.assertEqual(JwC2.shape, (self.nT,self.Np, self.nD, self.nD))

        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(ESS, tf.Variable))
        self.assertTrue(isinstance(Weights, tf.Variable))
        self.assertTrue(isinstance(JxC, tf.Variable))
        self.assertTrue(isinstance(JwC, tf.Variable))
        
        self.assertTrue(isinstance(X_filtered2, tf.Variable))
        self.assertTrue(isinstance(ESS2, tf.Variable))    
        self.assertTrue(isinstance(Weights2, tf.Variable))
        self.assertTrue(isinstance(JxC2, tf.Variable))
        self.assertTrue(isinstance(JwC2, tf.Variable))    

        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Weights)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC)))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ESS2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Weights2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC2)))






class TestKernelPFF(unittest.TestCase): 

    def setUp(self):
        
        self.Np         = 10
        self.nD         = 3
        self.nT         = 5

        self.Xprev      = tf.random.normal((self.nD,), dtype=tf.float64)
        self.Y          = tf.random.normal((self.nD,), dtype=tf.float64)
       
        self.A          = tf.linalg.diag(tf.random.uniform((self.nD,), -0.85, 0.85, dtype=tf.float64))
        self.B          = tf.linalg.diag(tf.random.uniform((self.nD,), -10.0, 10.0, dtype=tf.float64))

        self.V          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))
        self.W          = tf.linalg.diag(tf.random.uniform((self.nD,), 1e-3, 2.0, dtype=tf.float64))

        self.u          = tf.eye(self.nD, dtype=tf.float64) * 1e-9
        
        self.mu0        = tf.zeros((self.nD,), dtype=tf.float64)
        self.Sigma0     = (self.V @ self.V) @ tf.linalg.inv(tf.eye(self.nD, dtype=tf.float64) - self.A @ self.A) 
        
        self.epsilon    = 0.1
        self.Nl         = int(1/self.epsilon)

        self.P, _       = SE_Cov_div(self.nD, tf.random.normal((self.nD,), dtype=tf.float64) )
        self.R          = tf.linalg.diag(tf.range(self.nD, dtype=tf.float64) + 1.0)
        self.I          = tf.eye(self.nD, dtype=tf.float64)
        
    def test_eq13(self):
        e1              = tf.ones((self.nD,), dtype=tf.float64)
        A               = Hu21eq13(e1, self.mu0, self.I, self.P, self.I, self.u)
        A2              = tf.linalg.matvec( tf.linalg.inv( self.P @ tf.transpose(self.P) + self.u), e1)
        self.assertTrue( np.allclose(A,A2) )
        
    def test_eq15(self):
        e1              = tf.ones((self.nD,), dtype=tf.float64)
        A               = Hu21eq15(e1, self.mu0, self.P, self.u)
        A2              = tf.linalg.matvec( tf.linalg.inv(self.P + self.u), e1 )
        
        self.assertTrue(isinstance(A, tf.Tensor))
        self.assertTrue( np.allclose(A,A2) )
        
    def test_kpff_lp(self):
        e0              = tf.zeros((self.Np,self.nD), dtype=tf.float64)
        lp, jxc, jwc    = KPFF_LP(self.Np, self.nD, e0, self.mu0, self.mu0, self.B, self.W, self.mu0, self.Sigma0, self.u)

        self.assertEqual(lp.shape, (self.Np,self.nD))
        self.assertTrue(isinstance(lp, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(lp)))
        
        self.assertEqual(jxc.shape, (self.Np, self.nD, self.nD))
        self.assertTrue(isinstance(jxc, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jxc)))
        
        self.assertEqual(jwc.shape, (self.Np, self.nD, self.nD))
        self.assertTrue(isinstance(jwc, tf.Variable))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(jwc)))
        
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
        
        _, Y                    = SVSSM(self.nT, self.nD, A=self.A, B=self.B, V=self.V, W=self.W)
        X_filtered, JxC, JwC    = KernelPFF(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, stepsize=self.epsilon)
        X_filtered2, JxC2, JwC2 = KernelPFF(y=Y, N=self.Np, A=self.A, B=self.B, V=self.V, W=self.W, stepsize=self.epsilon, method="scalar")

        self.assertEqual(X_filtered.shape, (self.nT,self.nD))
        self.assertEqual(JxC.shape, (self.nT, self.Nl, self.Np, self.nD, self.nD))
        self.assertEqual(JwC.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        
        self.assertEqual(X_filtered2.shape, (self.nT,self.nD))
        self.assertEqual(JxC2.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))
        self.assertEqual(JwC2.shape, (self.nT,self.Nl,self.Np, self.nD, self.nD))

        self.assertTrue(isinstance(X_filtered, tf.Variable))
        self.assertTrue(isinstance(JxC, tf.Variable))
        self.assertTrue(isinstance(JwC, tf.Variable))
        self.assertTrue(isinstance(X_filtered2, tf.Variable)) 
        self.assertTrue(isinstance(JxC2, tf.Variable))
        self.assertTrue(isinstance(JwC2, tf.Variable))
        
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X_filtered2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JxC2)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(JwC2)))

        
if __name__ == '__main__':
    unittest.main()
