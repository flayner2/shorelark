use rand::{Rng, RngCore};

pub struct Network {
    layers: Vec<Layer>,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }

    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }

    pub fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }
}

impl Neuron {
    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }

    pub fn random(rng: &mut dyn rand::RngCore, output_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod random {
        use super::*;
        use approx::assert_relative_eq;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);

            assert_relative_eq!(neuron.bias, -0.6255188);
            assert_relative_eq!(
                neuron.weights.as_slice(),
                &[0.67383957, 0.8181262, 0.26284897, 0.5238807,].as_ref()
            );
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            // Ensures `.max()` (our ReLU) works:
            approx::assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

            // `0.5` and `1.0` chose by a fair dice roll:
            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
            );

            // We could've written `1.15` right away, but showing the entire
            // formula makes our intentions clearer
        }
    }

    mod layer {
        use super::*;

        mod propagate {
            use super::*;

            #[test]
            fn test() {
                let neuron1 = Neuron {
                    bias: 0.5,
                    weights: vec![-0.3, 0.8],
                };

                let neuron2 = Neuron {
                    bias: 0.4,
                    weights: vec![-0.1, 0.4],
                };

                let layer = Layer {
                    neurons: vec![neuron1, neuron2],
                };

                let result = layer.propagate(vec![-10.0, -10.0]);

                // Checking that ReLU does its thing
                approx::assert_relative_eq!(result.as_slice(), &[0.0, 0.0].as_ref());

                // Testing the real thing
                let result = layer.propagate(vec![0.5, 1.0]);

                // First neuron's output
                approx::assert_relative_eq!(result[0], (-0.3 * 0.5) + (0.8 * 1.0) + 0.5);
                // Second neuron's output
                approx::assert_relative_eq!(result[1], (-0.1 * 0.5) + (0.4 * 1.0) + 0.4);
            }
        }

        mod random {
            use super::*;
            use approx::assert_relative_eq;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());
                let layer = Layer::random(&mut rng, 2, 2);

                // Neuron biases
                assert_relative_eq!(layer.neurons[0].bias, -0.6255188);
                assert_relative_eq!(layer.neurons[1].bias, 0.26284897);

                // Neuron weights
                assert_relative_eq!(
                    layer.neurons[0].weights.as_slice(),
                    &[0.67383957, 0.8181262].as_ref()
                );
                assert_relative_eq!(
                    layer.neurons[1].weights.as_slice(),
                    &[0.5238807, -0.53516835].as_ref()
                );
            }
        }
    }

    mod network {
        use super::*;

        mod random {
            use super::*;
            use approx::assert_relative_eq;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());

                let layer1 = LayerTopology { neurons: 3 };
                let layer2 = LayerTopology { neurons: 2 };
                let layer3 = LayerTopology { neurons: 1 };
                let network = Network::random(&mut rng, &[layer1, layer2, layer3]);
                assert_eq!(network.layers.len(), 2);

                // Testing layer 1
                let n_layer1 = &network.layers[0];
                assert_eq!(n_layer1.neurons.len(), 2);
                // Neuron 1
                assert_relative_eq!(n_layer1.neurons[0].bias, -0.6255188);
                assert_relative_eq!(
                    n_layer1.neurons[0].weights.as_slice(),
                    &[0.67383957, 0.8181262, 0.26284897].as_ref()
                );
                // Neuron 2
                assert_relative_eq!(n_layer1.neurons[1].bias, 0.5238807);
                assert_relative_eq!(
                    n_layer1.neurons[1].weights.as_slice(),
                    &[-0.53516835, 0.069369674, -0.7648182].as_ref()
                );

                // Testing layer 2
                let n_layer2 = &network.layers[1];
                assert_eq!(n_layer2.neurons.len(), 1);
                assert_relative_eq!(n_layer2.neurons[0].bias, -0.102499366);
                assert_relative_eq!(
                    n_layer2.neurons[0].weights.as_slice(),
                    &[-0.48879617, -0.19277132].as_ref()
                );
            }
        }

        mod propagate {
            use super::*;
            use approx::assert_relative_eq;

            #[test]
            fn test() {
                let neuron1 = Neuron {
                    bias: 0.5,
                    weights: vec![-0.3, 0.8],
                };

                let neuron2 = Neuron {
                    bias: 0.4,
                    weights: vec![-0.1, 0.4],
                };

                let neuron3 = Neuron {
                    bias: -0.3,
                    weights: vec![0.4, 0.9],
                };

                let neuron4 = Neuron {
                    bias: -1.0,
                    weights: vec![-0.2, 0.6, 0.1],
                };

                let neuron5 = Neuron {
                    bias: -0.1,
                    weights: vec![0.7, -0.1, 0.4],
                };

                let neuron6 = Neuron {
                    bias: -0.9,
                    weights: vec![0.4, 0.2],
                };

                let layer1 = Layer {
                    neurons: vec![neuron1, neuron2, neuron3],
                };

                let layer2 = Layer {
                    neurons: vec![neuron4, neuron5],
                };

                let layer3 = Layer {
                    neurons: vec![neuron6],
                };

                let network = Network {
                    layers: vec![layer1, layer2, layer3],
                };

                // ReLU sanity check
                let result = network.propagate(vec![-10.0, -10.0]);
                assert_relative_eq!(result.as_slice(), &[0.0].as_ref());

                // Testing the propagate function for real
                // We calculate the output of each neuron separately for convenience
                // And calculate the final network output to compare against the actual output
                let result = network.propagate(vec![70.0, 60.0]);
                let r_n1 = f32::max((70.0 * -0.3) + (60.0 * 0.8) + 0.5, 0.0);
                let r_n2 = f32::max((70.0 * -0.1) + (60.0 * 0.4) + 0.4, 0.0);
                let r_n3 = f32::max((70.0 * 0.4) + (60.0 * 0.9) + -0.3, 0.0);
                let r_n4 = ((r_n1 * -0.2) + (r_n2 * 0.6) + (r_n3 * 0.1) - 1.0).max(0.0);
                let r_n5 = ((r_n1 * 0.7) + (r_n2 * -0.1) + (r_n3 * 0.4) - 0.1).max(0.0);
                let expected = &[((r_n4 * 0.4) + (r_n5 * 0.2) - 0.9).max(0.0)];
                assert_relative_eq!(result.as_slice(), expected.as_ref());
            }
        }
    }
}
