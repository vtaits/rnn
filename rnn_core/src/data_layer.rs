use std::sync::{Arc, RwLock};

use crate::Network;

pub struct DataLayerParams<T> {
    pub binary_to_data: Box<dyn Fn(&[bool]) -> Result<T, ()> + Send + Sync>,
    pub data_to_binary: Box<dyn Fn(T) -> Result<Vec<bool>, ()> + Send + Sync>,
    pub get_target_mask: Box<dyn Fn() -> Vec<bool> + Send + Sync>,
}

pub struct DataLayer<T> {
    params: DataLayerParams<T>,
    network: Arc<RwLock<Network>>,
    target_mask: Vec<bool>,
}

impl<T> DataLayer<T> {
    pub fn new(params: DataLayerParams<T>, network: Arc<RwLock<Network>>) -> Self {
        let target_mask = (params.get_target_mask)();

        DataLayer {
            params,
            network,
            target_mask,
        }
    }

    pub fn get_network(&self) -> Arc<RwLock<Network>> {
        Arc::clone(&self.network)
    }

    pub fn replace_network(&mut self, network: Network) {
        let mut current_network = self.network.write().unwrap();
        *current_network = network;
    }

    pub fn push_data_binary_and_apply(&mut self, bit_vec: &[bool], prediction_depth: usize) {
        self.network
            .write()
            .unwrap()
            .push_data_and_apply(bit_vec, prediction_depth);
    }

    fn check(&mut self, bit_vec: &[bool]) -> (bool, usize, usize) {
        let mut check_vec: Vec<bool> = vec![false; self.target_mask.len()];

        let mut false_positive_neurons = 0usize;
        let mut false_negative_neurons = 0usize;

        for (index, is_target) in self.target_mask.iter().enumerate() {
            let value = bit_vec[index];

            if !*is_target && value {
                check_vec[index] = true;
            }
        }

        let prediction = self.predict_binary(&check_vec, 0);

        let mut has_error = false;

        for (index, is_target) in self.target_mask.iter().enumerate() {
            if *is_target {
                let expected = bit_vec[index];
                let received = prediction[index];

                // print!("{}{} ", if expected {"+"} else {"."}, if received {"+"} else {"."});

                if !expected && received {
                    false_positive_neurons += 1;
                    has_error = true;
                }

                if expected && !received {
                    false_negative_neurons += 1;
                    has_error = true;
                }
            }
        }

        // println!();

        (!has_error, false_positive_neurons, false_negative_neurons)
    }

    pub fn count_accuracy(
        &mut self,
        measurement_data: Vec<Vec<bool>>,
    ) -> (usize, usize, usize, usize) {
        let mut positive = 0usize;
        let mut negative = 0usize;

        let mut false_positive_neurons = 0usize;
        let mut false_negative_neurons = 0usize;

        for item in measurement_data.iter() {
            let (is_positive, false_positive_neurons_result, false_negative_neurons_result) =
                self.check(&item);

            if is_positive {
                positive += 1;
            } else {
                negative += 1;
                false_positive_neurons += false_positive_neurons_result;
                false_negative_neurons += false_negative_neurons_result;
            }
        }

        (
            positive,
            negative,
            false_positive_neurons,
            false_negative_neurons,
        )
    }

    pub fn process_for_measure(&mut self, data: T) -> Result<Vec<bool>, ()> {
        (self.params.data_to_binary)(data)
    }

    pub fn push_data_and_apply(&mut self, data: T, prediction_depth: usize) {
        let bit_vec_result = (self.params.data_to_binary)(data);

        if let Ok(bit_vec) = bit_vec_result {
            self.push_data_binary_and_apply(&bit_vec, prediction_depth);
        }
    }

    pub fn push_data(&mut self, data: T, prediction_depth: usize) {
        let bit_vec_result = (self.params.data_to_binary)(data);

        if let Ok(bit_vec) = bit_vec_result {
            self.network
                .write()
                .unwrap()
                .push_data_binary(&bit_vec, prediction_depth);
        }
    }

    pub fn predict_binary(&mut self, bit_vec: &[bool], prediction_depth: usize) -> Vec<bool> {
        let binary_result = self
            .network
            .write()
            .unwrap()
            .predict(&bit_vec, prediction_depth);

        binary_result
    }

    pub fn deserialize(&mut self, binary_result: &[bool]) -> Result<T, ()> {
        let data_result = (self.params.binary_to_data)(binary_result);

        data_result
    }

    pub fn predict(&mut self, data: T, prediction_depth: usize) -> Vec<bool> {
        let bit_vec = (self.params.data_to_binary)(data).unwrap();

        self.predict_binary(&bit_vec, prediction_depth)
    }

    pub fn predict_and_deserialize(&mut self, data: T, prediction_depth: usize) -> Result<T, ()> {
        let binary_result = self.predict(data, prediction_depth);

        self.deserialize(&binary_result)
    }
}
