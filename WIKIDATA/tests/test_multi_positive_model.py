import unittest

try:
    import torch
    import torch.nn.functional as F
    from WIKIDATA.dart_encoder.model import multi_positive_contrastive_loss
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "A working PyTorch runtime is required")
class MultiPositiveLossTests(unittest.TestCase):
    def test_single_positive_matches_legacy_in_batch_cross_entropy(self):
        query = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
        positives = query.clone()
        expected = F.cross_entropy(query @ positives.T / 0.1, torch.arange(2))
        actual = multi_positive_contrastive_loss(
            query, positives, [1, 1], [["Q1"], ["Q2"]], 0.1
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_alternate_positive_is_rewarded_not_treated_as_negative(self):
        query = F.normalize(
            torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), dim=-1
        )
        positives = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]]), dim=-1)
        loss = multi_positive_contrastive_loss(
            query, positives, [2, 1], [["Q1", "Q2"], ["Q3"]], 0.1,
            positive_qids_flat=["Q1", "Q2", "Q3"],
        )
        self.assertLess(loss.item(), 0.001)

    def test_loss_backpropagates_with_query_specific_hard_negatives(self):
        query = F.normalize(torch.tensor([[1.0, 0.2]], requires_grad=True), dim=-1)
        positives = F.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2]]), dim=-1)
        negatives = F.normalize(torch.tensor([[0.0, 1.0]]), dim=-1)
        loss = multi_positive_contrastive_loss(
            query, positives, [2], [["Q1", "Q2"]], 0.1,
            positive_qids_flat=["Q1", "Q2"],
            neg_emb=negatives,
            neg_counts=[1],
        )
        loss.backward()
        self.assertIsNotNone(query.grad_fn)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
