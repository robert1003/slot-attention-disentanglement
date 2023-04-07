import torch

"""Implementation of the adjusted Rand index."""

def adjusted_rand_index(true_mask, pred_mask):
    """Computes the adjusted Rand index (ARI), a clustering similarity score.
    
    Args:
        true_mask: `Tensor` of shape [batch_size, n_points, one_hot_groups].
        The true cluster assignment encoded as one-hot.
        pred_mask: `Tensor` of shape [batch_size, n_points, preds].
        The predicted cluster assignment encoded as categorical probabilities.
        This function works on the argmax over axis 2.
    
    Returns:
        ARI scores as a `Tensor` of shape [batch_size].
    
    Raises:
        ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
        The special cases that can occur when you have
        one cluster per datapoint is not handled.
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
        # This rules out the n_true_groups == n_pred_groups == n_points
        # corner case, and also n_true_groups == n_pred_groups == 0, since
        # that would imply n_points == 0 too.
        # The sklearn implementation has a corner-case branch which does
        # handle this. We chose not to support these cases to avoid counting
        # distinct clusters just to check if we have one cluster per datapoint.
        raise ValueError(
            "adjusted_rand_index requires n_groups < n_points. We don't handle "
            "the special cases that can occur when you have one cluster "
            "per datapoint.")
    
    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    # We convert true and predicted clusters to one-hot ('oh') representations.
    pred_mask_oh = torch.nn.functional.one_hot(pred_group_ids, n_pred_groups).float()
    true_mask = true_mask.float()
    # Sum along dim 1 and 2 to get the number of points in each cluster.
    n_points = torch.sum(true_mask, dim=(1, 2))
    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=(1, 2))
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # The case where n_true_groups == n_pred_groups == 1 needs to be
    # special-cased (to return 1) as the above formula gives a divide-by-zero.
    # This might not work when true_mask has values that do not sum to one:

    both_single_cluster = torch.logical_and(
    _all_equal(true_group_ids), _all_equal(pred_group_ids))
    result = torch.where(both_single_cluster, torch.ones_like(ari), ari)
    return result


def _all_equal(values):
  """Whether values are all equal along the final axis."""
  return torch.all(torch.eq(values, values[..., :1]), dim=-1)

if __name__ == "__main__":
    # true_mask = torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 1], [0,0,1],[0, 1, 0], [1, 0, 0], [0, 0, 1], [0,0,1]]])
    # pred_mask = torch.tensor([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.2, 0.6],[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.5, 0.1, 0.4]]])
    true_mask = torch.tensor([[[1, 0, 0], [1, 0, 0], [0, 1, 0], [0,1,0]]])
    pred_mask = torch.tensor([[[1,0,0], [1, 0, 0], [0, 1, 0], [0, 0, 2]]])
    print(adjusted_rand_index(true_mask, pred_mask))
