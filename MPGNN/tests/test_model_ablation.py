import unittest

import torch

from Model import NodeMLP, PaperActor, PaperCritic


class NodeMLPTest(unittest.TestCase):
    def test_single_graph_shape_and_gradient(self):
        model = NodeMLP(5, 8, 1, depth=3)
        inputs = torch.randn(7, 5, requires_grad=True)
        output = model(inputs, torch.tensor([[0, 1], [1, 2]]))
        self.assertEqual(tuple(output.shape), (7, 1))
        output.sum().backward()
        self.assertIsNotNone(inputs.grad)

    def test_batch_shape_and_hidden_states(self):
        model = NodeMLP(4, 6, 1, depth=2)
        inputs = torch.randn(3, 9, 4)
        output, hidden = model.forward_with_hidden(inputs, None)
        self.assertEqual(tuple(output.shape), (3, 9, 1))
        self.assertEqual(len(hidden), 1)
        self.assertEqual(tuple(hidden[0].shape), (3, 9, 6))

    def test_paper_actor_and_critic_shapes(self):
        edge_index = torch.tensor([[0, 1, 2, 3, 1, 2],
                                    [1, 2, 3, 0, 0, 1]])
        obs = torch.randn(4, 4, requires_grad=True)
        actor = PaperActor(input_dim=4, gat_hidden=8, actor_hidden=16,
                           gat_depth=3, gat_heads=4)
        actions = actor(obs, edge_index)
        self.assertEqual(tuple(actions.shape), (4, 1))
        self.assertTrue(bool(torch.all(actions >= 0)))
        self.assertTrue(bool(torch.all(actions <= 1)))

        critic = PaperCritic(input_dim=5, hidden_dim=16)
        values = critic(torch.cat((obs, actions), dim=-1), edge_index)
        self.assertEqual(tuple(values.shape), (4, 1))
        values.sum().backward()
        self.assertIsNotNone(obs.grad)


if __name__ == "__main__":
    unittest.main()
