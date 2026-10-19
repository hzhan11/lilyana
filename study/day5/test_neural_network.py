import unittest

import numpy as np

from train_model import NeuralNetwork


class NeuralNetworkTests(unittest.TestCase):
    def test_shapes_and_probabilities(self):
        model = NeuralNetwork()
        hidden, log_probs = model.forward(np.ones((3, 4096)))
        self.assertEqual(model.weights1.shape, (4096, 32))
        self.assertEqual(model.weights2.shape, (32, 10))
        self.assertEqual(hidden.shape, (3, 32))
        self.assertEqual(log_probs.shape, (3, 10))
        self.assertTrue((hidden >= 0).all())
        np.testing.assert_allclose(np.exp(log_probs).sum(axis=1), 1)

    def test_gradients_for_both_layers(self):
        rng = np.random.default_rng(7)
        model = NeuralNetwork(input_size=4, hidden_size=5, output_size=3)
        x = rng.normal(size=(6, 4))
        y = np.array([0, 1, 2, 0, 1, 2])
        parameters = (model.weights1, model.bias1, model.weights2, model.bias2)
        epsilon = 1e-6
        for parameter, gradient in zip(parameters, model.gradients(x, y)):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = model.evaluate(x, y)[0]
                parameter[index] = original - epsilon
                minus = model.evaluate(x, y)[0]
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=6)

    def test_training_updates_both_layers_and_evaluation_is_read_only(self):
        model = NeuralNetwork(input_size=2, hidden_size=8, output_size=2)
        x = np.array([[1.0, 0], [0, 1.0]])
        y = np.array([0, 1])
        parameters = (model.weights1, model.bias1, model.weights2, model.bias2)
        before = [parameter.copy() for parameter in parameters]
        initial_loss = model.evaluate(x, y)[0]
        for _ in range(100):
            model.train_step(x, y, 0.1)
        self.assertLess(model.evaluate(x, y)[0], initial_loss)
        for old, current in zip(before, parameters):
            self.assertTrue(np.any(current != old))
        after_training = [parameter.copy() for parameter in parameters]
        model.evaluate(x, y)
        for old, current in zip(after_training, parameters):
            np.testing.assert_array_equal(old, current)


if __name__ == "__main__":
    unittest.main()
