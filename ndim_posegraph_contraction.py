"""
N-dim Pose Graph Estimation.

Linear problem where we are estimating the position in N-dim space of a number of nodes.
Linear factors connect each node to the M closest nodes in the space.
The linear factors measure the distance between the nodes in each of the N dimensions.
"""

import copy
from matplotlib import animation, pyplot as plt
import numpy as np
import argparse

from gbp import gbp
from gbp.factors import linear_displacement
from itertools import combinations
from collections import defaultdict

np.random.seed(0)

def generate_duplicate_node_ids(measurements_nodeIDs):
    
    # Step 1: Count how many times each node appears (this counts the factors for each node)
    node_factor_count = defaultdict(int)
    for node1, node2 in measurements_nodeIDs:
        node_factor_count[node1] += 1
        node_factor_count[node2] += 1

    # Step 2: Generate unique duplicate node IDs
    duplicate_node_ids = {}
    unique_id_counter = defaultdict(int)  # To ensure uniqueness of the generated duplicate IDs
    
    for node_id, count in node_factor_count.items():
        for i in range(count):
            # Generate a new unique duplicate for each factor (node ID)
            # duplicate_id = f"{node_id}_{unique_id_counter[node_id]}"
            duplicate_id = unique_id_counter[node_id] + node_id * 1000
            if node_id not in duplicate_node_ids:
                duplicate_node_ids[node_id] = [duplicate_id]
            else:
                duplicate_node_ids[node_id].append(duplicate_id)
            unique_id_counter[node_id] += 1

    return duplicate_node_ids


parser = argparse.ArgumentParser()
parser.add_argument("--n_varnodes", type=int, default=50,
                    help="Number of variable nodes.")
parser.add_argument("--dim", type=int, default=6,
                    help="Dimensionality of space nodes exist in (dofs of variables)")
parser.add_argument("--M", type=int, default=10,
                    help="Each node is connected to its k closest neighbours by a measurement.")
parser.add_argument("--gauss_noise_std", type=float, default=1.,
                    help="Standard deviation of Gaussian noise added to measurement model (pixels)")

parser.add_argument("--n_iters", type=int, default=50,
                    help="Number of iterations of GBP")

args = parser.parse_args()
print('Configs: \n', args)


# Create priors
priors_mu = np.random.rand(args.n_varnodes, args.dim) * 10  # grid goes from 0 to 10 along x and y axis
prior_sigma = 3 * np.eye(args.dim)

prior_lambda = np.linalg.inv(prior_sigma)
priors_lambda = [prior_lambda] * args.n_varnodes
priors_eta = []
for mu in priors_mu:
    priors_eta.append(np.dot(prior_lambda, mu))

# Generate connections between variables
gt_measurements, noisy_measurements = [], []
measurements_nodeIDs = []
num_edges_per_node = np.zeros(args.n_varnodes)
n_edges = 0

for i, mu in enumerate(priors_mu):
    dists = []
    for j, mu1 in enumerate(priors_mu):
        dists.append(np.linalg.norm(mu - mu1))
    for j in np.array(dists).argsort()[1:args.M + 1]:  # As closest node is itself
        mu1 = priors_mu[j]
        if [j, i] not in measurements_nodeIDs:  # To avoid double counting
            n_edges += 1
            gt_measurements.append(mu - mu1)
            noisy_measurements.append(mu - mu1 + np.random.normal(0., args.gauss_noise_std, args.dim))
            measurements_nodeIDs.append([i, j])

            num_edges_per_node[i] += 1
            num_edges_per_node[j] += 1

print(f"measurements ", measurements_nodeIDs)


graph = gbp.FactorGraph(nonlinear_factors=False)

# Initialize variable nodes for frames with prior
new_ids_dict = generate_duplicate_node_ids(measurements_nodeIDs)
dummy_new_ids = copy.deepcopy(new_ids_dict)
print(new_ids_dict)
# iterative add ids in new_ids
graph_vars = {}
for id, new_ids in new_ids_dict.items():
    print (f"new_ids {new_ids}")
    for new_id in new_ids:
        new_var_node = gbp.VariableNode(new_id, args.dim)
        new_var_node.prior.eta = priors_eta[id]*2
        new_var_node.prior.lam = priors_lambda[id]*100
        graph.var_nodes.append(new_var_node)
        if new_id not in graph_vars:
            # index of the new node in the graph.var_nodes
            graph_vars[new_id] = len(graph.var_nodes) - 1
        else:
            assert False, f"new_id {new_id} already exists in graph_vars"

num_factor = len(noisy_measurements)
for f, measurement in enumerate(noisy_measurements):

    id1 = measurements_nodeIDs[f][0]
    id2 = measurements_nodeIDs[f][1]
    new_id1 = dummy_new_ids[id1].pop(0)
    new_id2 = dummy_new_ids[id2].pop(0)
    factor_id = f
    new_factor = gbp.Factor(factor_id,
                            [graph.var_nodes[graph_vars[new_id1]], graph.var_nodes[graph_vars[new_id2]]],
                            measurement,
                            args.gauss_noise_std,
                            linear_displacement.meas_fn,
                            linear_displacement.jac_fn,
                            loss=None,
                            mahalanobis_threshold=2)
    graph.var_nodes[graph_vars[new_id1]].adj_factors.append(new_factor)
    graph.var_nodes[graph_vars[new_id2]].adj_factors.append(new_factor)
    graph.factors.append(new_factor)

# check if all the new_ids are used
for id, new_ids in dummy_new_ids.items():
    assert len(new_ids) == 0, f"new_ids {new_ids} are not used"
# add contraction factors

# factor_id
for id, new_ids in new_ids_dict.items():
    pairs = list(combinations(new_ids, 2))
    for pair in pairs:
        factor_id += 1
        new_factor = gbp.ContractionFactor(factor_id,
                                [graph.var_nodes[graph_vars[pair[0]]], graph.var_nodes[graph_vars[pair[1]]]],
                                np.zeros(args.dim),
                                args.gauss_noise_std,
                                loss=None,
                                Nstds=2)
        graph.var_nodes[graph_vars[pair[0]]].adj_factors.append(new_factor)
        graph.var_nodes[graph_vars[pair[1]]].adj_factors.append(new_factor)
        graph.factors.append(new_factor)
        num_factor += 1


graph.update_all_beliefs()
graph.compute_all_factors()

graph.n_var_nodes = len(graph.var_nodes)
graph.n_factor_nodes = len(noisy_measurements)
graph.n_edges = 2 * len(noisy_measurements)

print(f'Number of variable nodes {graph.n_var_nodes}')
print(f'Number of edges per variable node {args.M}')
print(f'Number of dofs at each variable node {args.dim}\n')

# mu, sigma = graph.joint_distribution_cov()  # Get batch solution

fig, ax = plt.figure(), plt.gca()

def update(frame):
    graph.synchronous_iteration()
    ax.clear()
    for var_node in graph.var_nodes:
        ax.scatter(var_node.belief.mu[0], var_node.belief.mu[1], c='r')
    ax.set_title(f'Iteration {frame}   //   Energy {graph.energy():.4f}')
    print(f'Iteration {frame}   //   Energy {graph.energy():.4f}')

ani = animation.FuncAnimation(fig, update, frames=args.n_iters, repeat=False)

# Save the animation as a video file
ani.save('posegraph_contraction.mp4', writer='ffmpeg', fps=1)

plt.show()