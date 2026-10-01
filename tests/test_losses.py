import pytest
import torch
from torch import nn

from setfit.losses import SINCERELoss, SupConLoss


class _IdentityBody(nn.Module):
    """Stands in for a Sentence Transformer: returns the given features as the embedding."""

    def forward(self, features):
        return {"sentence_embedding": features["x"]}


def _loss(loss_class, embeddings, labels, **kwargs):
    return loss_class(_IdentityBody(), **kwargs)([{"x": embeddings}], labels)


def _naive_sincere(embeddings, labels, temperature):
    """Direct transcription of the SINCERE loss (Feeney & Hughes, arXiv 2309.14277, eq. 6):
    for each anchor i and each positive p != i of the same class,
    -log( exp(s_ip) / (exp(s_ip) + sum_{n: y_n != y_i} exp(s_in)) ),
    averaged over the positives of each anchor, then over anchors that have a positive."""
    z = torch.nn.functional.normalize(embeddings, dim=1)
    sim = z @ z.T / temperature
    per_anchor = []
    for i in range(len(labels)):
        positives = [p for p in range(len(labels)) if p != i and labels[p] == labels[i]]
        negatives = [n for n in range(len(labels)) if labels[n] != labels[i]]
        if not positives:
            continue
        terms = []
        for p in positives:
            denom = torch.exp(sim[i, p]) + sum((torch.exp(sim[i, n]) for n in negatives), torch.tensor(0.0))
            terms.append(-torch.log(torch.exp(sim[i, p]) / denom))
        per_anchor.append(torch.stack(terms).mean())
    return torch.stack(per_anchor).mean()


@pytest.mark.parametrize("temperature", [0.07, 0.5, 1.0])
@pytest.mark.parametrize("seed", range(5))
def test_sincere_matches_naive_formula(seed, temperature):
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(12, 8, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 3, (12,), generator=generator)
    expected = _naive_sincere(embeddings, labels, temperature)
    actual = _loss(SINCERELoss, embeddings, labels, temperature=temperature)
    torch.testing.assert_close(actual, expected)


def test_sincere_equals_supcon_with_one_positive_per_anchor():
    # With exactly two samples per class, each anchor has one positive, so the SupCon denominator
    # contains no other same-class sample and the two losses must agree.
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    torch.testing.assert_close(_loss(SINCERELoss, embeddings, labels), _loss(SupConLoss, embeddings, labels))


def test_sincere_differs_from_supcon_with_several_positives_per_anchor():
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(9, 16, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert not torch.allclose(_loss(SINCERELoss, embeddings, labels), _loss(SupConLoss, embeddings, labels))


def _supcon_from_logits(logits, labels):
    """SupCon L_out written on a similarity matrix (Khosla et al., arXiv 2004.11362, eq. 2)."""
    n = len(labels)
    not_self = ~torch.eye(n, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & not_self
    log_prob = logits - torch.logsumexp(logits.masked_fill(~not_self, float("-inf")), dim=1, keepdim=True)
    return -((log_prob * positives).sum(1) / positives.sum(1)).mean()


@pytest.mark.parametrize("seed", range(5))
def test_sincere_never_repels_same_class_samples(seed):
    # Intra-class repulsion = a positive gradient of the loss with respect to the similarity of two
    # same-class samples (increasing that similarity would increase the loss). For anchor i and positive q,
    # SupCon's gradient on s_iq is softmax_iq - 1/|P_i|, so it repels any positive that already takes more
    # than its 1/|P_i| share of the softmax. Sample 1 is made that close to anchor 0 below.
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(len(labels), len(labels), generator=generator, dtype=torch.float64)
    logits = (logits + logits.T) / 2
    logits[0, 1] = logits[1, 0] = 6.0
    logits.requires_grad_(True)
    same_class = (labels[:, None] == labels[None, :]) & ~torch.eye(len(labels), dtype=torch.bool)

    SINCERELoss.loss_from_logits(logits, labels).backward()
    assert (logits.grad[same_class] <= 0).all()

    logits.grad = None
    _supcon_from_logits(logits, labels).backward()
    assert logits.grad[0, 1] > 0


def test_sincere_ignores_anchors_without_positives():
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(5, 8, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 2])  # label 2 has no positive in the batch
    loss = _loss(SINCERELoss, embeddings, labels)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, _naive_sincere(embeddings, labels, 0.07))


def test_sincere_is_zero_without_negatives():
    # A batch with a single class has an empty negative set: every positive term is -log(1) = 0.
    embeddings = torch.randn(4, 8, dtype=torch.float64)
    loss = _loss(SINCERELoss, embeddings, torch.tensor([1, 1, 1, 1]))
    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64))


def test_sincere_is_finite_for_extreme_similarities():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    loss = _loss(SINCERELoss, embeddings, torch.tensor([0, 0, 1, 1]), temperature=1e-3)
    assert torch.isfinite(loss)


def test_sincere_backpropagates_to_embeddings():
    embeddings = torch.randn(6, 8, requires_grad=True)
    _loss(SINCERELoss, embeddings, torch.tensor([0, 0, 1, 1, 2, 2])).backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
