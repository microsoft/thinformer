import numpy as np
from scipy import linalg
import warnings
from typing import Iterable, Optional, Tuple
import sys

class FIDStatistics:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        self.mu = mu
        self.sigma = sigma

    def frechet_distance(self, other, eps=1e-6):
        """
        Compute the Frechet distance between two sets of statistics.
        """
        # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
        mu1, sigma1 = self.mu, self.sigma
        mu2, sigma2 = other.mu, other.sigma

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), f"Training and test mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
        assert (
            sigma1.shape == sigma2.shape
        ), f"Training and test covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; adding %s to diagonal of cov estimates"
                % eps
            )
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    

def read_statistics(
    npz_path: str) -> Tuple[FIDStatistics, FIDStatistics]:
    obj = np.load(npz_path)
    if "mu" in list(obj.keys()):
        return FIDStatistics(obj["mu"], obj["sigma"]), FIDStatistics(
            obj["mu_s"], obj["sigma_s"]
        )
    
def compute_statistics(self, activations: np.ndarray) -> FIDStatistics:
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return FIDStatistics(mu, sigma)


obj = np.load("./examples/bigGAN/utils/VIRTUAL_imagenet512.npz")

mu, sigma = read_statistics(npz_path = "./examples/bigGAN/utils/VIRTUAL_imagenet512.npz")
print(mu, sigma)