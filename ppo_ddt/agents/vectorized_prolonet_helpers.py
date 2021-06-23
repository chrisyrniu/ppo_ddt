# Created by Andrew Silva on 2/21/19
import numpy as np
import sys
sys.path.insert(0, '../')
from agents.vectorized_prolonet import ProLoNet
import torch


def add_level(pro_lo_net, split_noise_scale=0.2, use_gpu=False):
    old_weights = pro_lo_net.layers  # Get the weights out
    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]

    old_comparators = pro_lo_net.comparators  # get the comparator values out
    new_comparators = [comp.detach().clone().data.cpu().numpy() for comp in
                       old_comparators]
    old_leaf_information = pro_lo_net.leaf_init_information  # get the leaf init info out

    new_leaf_information = []

    for leaf_index in range(len(old_leaf_information)):
        prior_leaf = pro_lo_net.action_probs[leaf_index].detach().clone().cpu().numpy()
        leaf_information = old_leaf_information[leaf_index]
        left_path = leaf_information[0]
        right_path = leaf_information[1]
        # This hideousness is to handle empty sequences... get the index of the node this used to split at
        weight_index = max(
            max(max(left_path,
                    [-1])
                ),
            max(max(right_path,
                    [-1])
                )
        )

        new_weight = np.random.normal(scale=split_noise_scale,
                                      size=old_weights[weight_index].size()[0])
        new_comparator = np.random.normal(scale=split_noise_scale,
                                          size=old_comparators[weight_index].size()[0])

        new_weights.append(new_weight)  # Add it to the list of nodes
        new_comparators.append(new_comparator)  # Add it to the list of nodes

        new_node_ind = len(new_weights) - 1  # Remember where we put it

        # Create our two new leaves
        new_leaf1 = np.random.normal(scale=split_noise_scale, size=prior_leaf.shape)
        new_leaf2 = np.random.normal(scale=split_noise_scale, size=prior_leaf.shape)

        # Create the paths, which are copies of the old path but now with a left / right at the new node
        new_leaf1_left = left_path.copy()
        new_leaf1_right = right_path.copy()
        new_leaf2_left = left_path.copy()
        new_leaf2_right = right_path.copy()
        # Leaf 1 goes left at the new node, leaf 2 goes right
        new_leaf1_left.append(new_node_ind)
        new_leaf2_right.append(new_node_ind)

        new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
        new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])

        new_network = ProLoNet(input_dim=pro_lo_net.input_dim, weights=new_weights, comparators=new_comparators,
                               leaves=new_leaf_information, alpha=pro_lo_net.alpha.item(), is_value=pro_lo_net.is_value,
                               device='cuda' if use_gpu else 'cpu', vectorized=pro_lo_net.vectorized,
                               output_dim=pro_lo_net.output_dim)
        if use_gpu:
            new_network = new_network.cuda()
    return new_network


def swap_in_node(network, deeper_network, leaf_index, use_gpu=False):
    """
    Duplicates the network and returns a new one, where the node at leaf_index as been turned into a splitting node
    with two leaves that are slightly noisy copies of the previous node
    :param network: prolonet in
    :param deeper_network: deeper_network to take the new node / leaves from
    :param leaf_index: index of leaf to turn into a split
    :return: new prolonet (value or normal)
    """
    old_weights = network.layers  # Get the weights out
    old_comparators = network.comparators  # get the comparator values out
    leaf_information = network.leaf_init_information[leaf_index]  # get the old leaf init info out
    left_path = leaf_information[0]
    right_path = leaf_information[1]
    if deeper_network is not None:
        deeper_weights = [weight.detach().clone().data.cpu().numpy() for weight in deeper_network.layers]

        deeper_comparators = [comp.detach().clone().data.cpu().numpy() for comp in
                              deeper_network.comparators]

        deeper_leaf_info = deeper_network.leaf_init_information[leaf_index*2]
        deeper_left_path = deeper_leaf_info[0]
        deeper_right_path = deeper_leaf_info[1]
        deeper_weight_index = max(
            max(max(deeper_left_path,
                    [-1])
                ),
            max(max(deeper_right_path,
                    [-1])
                )
        )

        # Make a new weight vector, mostly the same as the old one

        new_weight = deeper_weights[deeper_weight_index]
        new_comparator = deeper_comparators[deeper_weight_index]
        new_leaf1 = deeper_network.action_probs[leaf_index * 2].detach().clone().data.cpu().numpy()
        new_leaf2 = deeper_network.action_probs[leaf_index * 2 + 1].detach().clone().data.cpu().numpy()
    else:
        new_weight = np.random.normal(scale=0.2,
                                      size=old_weights[0].size()[0])
        new_comparator = np.random.normal(scale=0.2,
                                          size=old_comparators[0].size()[0])
        new_leaf1 = np.random.normal(scale=0.2,
                                     size=network.action_probs[leaf_index].size()[0])
        new_leaf2 = np.random.normal(scale=0.2,
                                     size=network.action_probs[leaf_index].size()[0])

    new_weights = [weight.detach().clone().data.cpu().numpy() for weight in old_weights]
    new_weights.append(new_weight)  # Add it to the list of nodes
    new_comparators = [comp.detach().clone().data.cpu().numpy() for comp in old_comparators]
    new_comparators.append(new_comparator)  # Add it to the list of nodes

    new_node_ind = len(new_weights) - 1  # Remember where we put it

    # Create the paths, which are copies of the old path but now with a left / right at the new node
    new_leaf1_left = left_path.copy()
    new_leaf1_right = right_path.copy()
    new_leaf2_left = left_path.copy()
    new_leaf2_right = right_path.copy()
    # Leaf 1 goes left at the new node, leaf 2 goes right
    new_leaf1_left.append(new_node_ind)
    new_leaf2_right.append(new_node_ind)

    new_leaf_information = network.leaf_init_information
    for index, leaf_prob_vec in enumerate(network.action_probs):  # Copy over the learned leaf weight
        new_leaf_information[index][-1] = leaf_prob_vec.detach().clone().data.cpu().numpy()
    new_leaf_information.append([new_leaf1_left, new_leaf1_right, new_leaf1])
    new_leaf_information.append([new_leaf2_left, new_leaf2_right, new_leaf2])
    # Remove the old leaf
    del new_leaf_information[leaf_index]
    new_network = ProLoNet(input_dim=network.input_dim, weights=new_weights, comparators=new_comparators,
                           leaves=new_leaf_information, alpha=network.alpha.item(), is_value=network.is_value,
                           device='cuda' if use_gpu else 'cpu',
                           vectorized=network.vectorized, output_dim=network.output_dim)
    if use_gpu:
        new_network = new_network.cuda()
    return new_network

def convert_to_crisp(fuzzy_model, training_data):
    new_weights = []
    new_comps = []
    device = fuzzy_model.device
    if not fuzzy_model.vectorized:
        weights = np.abs(fuzzy_model.layers.cpu().detach().numpy())
        most_used = np.argmax(weights, axis=1)
        for comp_ind, comparator in enumerate(fuzzy_model.comparators):
            comparator = comparator.item()
            divisor = abs(fuzzy_model.layers[comp_ind][most_used[comp_ind]].item())
            if divisor == 0:
                divisor = 1
            comparator /= divisor
            new_comps.append([comparator])
            max_ind = most_used[comp_ind]
            new_weight = np.zeros(len(fuzzy_model.layers[comp_ind].data))
            new_weight[max_ind] = fuzzy_model.layers[comp_ind][most_used[comp_ind]].item() / divisor
            new_weights.append(new_weight)
    else:
        # comp = comp.sub(fuzzy_model.comparators.expand(input_data.size(0), *fuzzy_model.comparators.size()))
        # sig_vals = fuzzy_model.sig(comp)
        # most_used = torch.argmax(sig_vals, dim=2).mode(dim=0)[0].tolist()
        most_used = torch.argmax(fuzzy_model.selector, dim=1).tolist()

        for index in range(len(fuzzy_model.layers)):
            # max_ind = np.argmax(acc_model.layers[index].data)
            max_ind = most_used[index]
            new_weight = np.zeros(len(fuzzy_model.layers[index].data))
            divisor = abs(fuzzy_model.layers[index][max_ind].item())
            if divisor == 0:
                divisor = 1
            new_weight[max_ind] = fuzzy_model.layers[index][max_ind].item()/divisor
            new_comparator = np.zeros(len(fuzzy_model.comparators[index].data))
            new_comparator[max_ind] = fuzzy_model.comparators[index][max_ind].data/divisor
            new_weights.append(new_weight)
            new_comps.append(new_comparator)

    new_input_dim = fuzzy_model.input_dim
    new_weights = np.array(new_weights)
    new_comps = np.array(new_comps)
    new_alpha = fuzzy_model.alpha.item()
    new_alpha = 99999. * new_alpha / np.abs(new_alpha)
    crispy_model = ProLoNet(input_dim=new_input_dim,
                            output_dim=fuzzy_model.output_dim,
                            weights=new_weights,
                            comparators=new_comps,
                            leaves=fuzzy_model.leaf_init_information,
                            alpha=new_alpha,
                            vectorized=fuzzy_model.vectorized,
                            device=device).to(device)
    crispy_model.action_probs.data = fuzzy_model.action_probs.data

    return crispy_model
