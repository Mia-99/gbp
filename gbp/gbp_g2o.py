"""
    Defines child classes of GBP parent classes for Bundle Adjustment.
    Also defines the function to create the factor graph.
"""

import numpy as np
from gbp import gbp, gbp_g2o
from gbp.factors import reprojection
from utils import read_g2ofile


class G2OFactorGraph(gbp.FactorGraph):
    def __init__(self, **kwargs):
        gbp.FactorGraph.__init__(self, nonlinear_factors=True, **kwargs)

        self.pose_nodes = []
        # self.lmk_nodes = []
        self.var_nodes = self.pose_nodes

    def generate_priors_var(self, weaker_factor=100):
        """
            Sets automatically the std of the priors such that standard deviations
            of prior factors are a factor of weaker_factor
            weaker than the standard deviations of the adjacent factors.
            NB. Jacobian of measurement function effectively sets the scale of the factors.
        """
        for var_node in self.cam_nodes + self.lmk_nodes:
            max_factor_lam = 0.
            for factor in var_node.adj_factors:
                if isinstance(factor, gbp_ba.ReprojectionFactor):
                    max_factor_lam = max(max_factor_lam, np.max(factor.factor.lam))
            lam_prior = np.eye(var_node.dofs) * max_factor_lam / (weaker_factor ** 2)
            var_node.prior.lam = lam_prior
            var_node.prior.eta = lam_prior @ var_node.mu

    def weaken_priors(self, weakening_factor):
        """
            Increases the variance of the priors by the specified factor.
        """
        for var_node in self.var_nodes:
            var_node.prior.eta *= weakening_factor
            var_node.prior.lam *= weakening_factor

    def set_priors_var(self, priors):
        """
            Means of prior have already been set when graph was initialised. Here we set the variance of the prior factors.
            priors: list of length number of variable nodes where each element is the covariance matrix of the prior
                    distribution for that variable node.
        """
        for v, var_node in enumerate(self.var_nodes):
            var_node.prior.lam = np.linalg.inv(priors[v])
            var_node.prior.eta = var_node.prior.lam @ var_node.mu

    def compute_residuals(self):
        residuals = []
        for factor in self.factors:
            if isinstance(factor, ReprojectionFactor):
                residuals += list(factor.compute_residual())
        return residuals

    def are(self):
        """
            Computes the Average Reprojection Error across the whole graph.
        """
        are = 0
        for factor in self.factors:
            if isinstance(factor, ReprojectionFactor):
                are += np.linalg.norm(factor.compute_residual())
        return are / len(self.factors)


class FrameVariableNode(gbp.VariableNode):
    def __init__(self, variable_id, dofs, c_id=None):
        gbp.VariableNode.__init__(self, variable_id, dofs)
        self.c_id = c_id


class ReprojectionFactor(gbp.Factor):
    def __init__(self, factor_id, adj_var_nodes, measurement, gauss_noise_std, loss, Nstds, K):

        gbp.Factor.__init__(self, factor_id, adj_var_nodes, measurement, gauss_noise_std,
                            reprojection.meas_fn, reprojection.jac_fn, loss, Nstds, K)

    def reprojection_err(self):
        """
            Returns the reprojection error at the factor in pixels.
        """
        return np.linalg.norm(self.compute_residual())

class BetweenFactor(gbp.Factor):
    def __init__(self, factor_id, adj_var_nodes, measurement, gauss_noise_std, loss, Nstds):

        gbp.Factor.__init__(self, factor_id, adj_var_nodes, measurement, gauss_noise_std,
                            reprojection.meas_fn, reprojection.jac_fn, loss)

    def err(self):
        """
            Returns the reprojection error at the factor in pixels.
        """
        return np.linalg.norm(self.compute_residual())


def create_g2o_graph(g2o_file, configs):
    """
        Create graph object from bal style file.
    """

    # read from g2o file
    n_poses, n_edges, measurements, poses_ID1s, poses_ID2s, infos = read_g2ofile.read_g2ofile(g2o_file)
    print(f'Number of poses: {n_poses}')
    print(f'Number of edges: {n_edges}')
    print(f'pose_ID1s: {poses_ID1s}')
    print(f'pose_ID2s: {poses_ID2s}')
    
    graph = G2OFactorGraph(eta_damping=configs['eta_damping'],
                          beta=configs['beta'],
                          num_undamped_iters=configs['num_undamped_iters'],
                          min_linear_iters=configs['min_linear_iters'])

    variable_id = 0
    factor_id = 0
    n_edges = 0

    poses_IDs = []
    [poses_IDs.append(id) for id in (poses_ID1s + poses_ID2s) if id not in poses_IDs]

    print(f'pose_IDs: {poses_IDs}')
    
    priors_mu = np.random.rand(n_poses, 6)  # grid goes from 0 to 10 along x and y axis
    prior_sigma = 3 * np.eye(6)


    for m, init_params in enumerate(poses_IDs):
        new_pose_node = FrameVariableNode(variable_id, 6, m)
        new_pose_node.mu = init_params
        graph.pose_nodes.append(new_pose_node)
        variable_id += 1

    for f, measurement in enumerate(measurements):
        print(f'f: {f}')
        pose_node1 = graph.pose_nodes[poses_ID1s[f]]
        pose_node2 = graph.pose_nodes[poses_ID2s[f]]
        info = infos[f]
        cov = np.linalg.inv(info)

        # Compute the standard deviation (square root of the diagonal elements of cov)
        gauss_noise_std = np.sqrt(np.diagonal(cov))

        new_factor = BetweenFactor(factor_id, [pose_node1, pose_node2], measurement,
                                   gauss_noise_std, configs['loss'], configs['Nstds'])

        linpoint = np.concatenate((pose_node1.mu, pose_node2.mu))
        new_factor.compute_factor(linpoint)
        pose_node1.adj_factors.append(new_factor)
        pose_node2.adj_factors.append(new_factor)

        graph.factors.append(new_factor)
        factor_id += 1
        n_edges += 1

    graph.n_factor_nodes = factor_id
    graph.n_var_nodes = variable_id
    graph.var_nodes = graph.cam_nodes + graph.lmk_nodes
    graph.n_edges = n_edges

    return graph


