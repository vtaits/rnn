use std::{sync::Arc, sync::RwLock};

use crate::Network;

pub struct DataLayerParams<T> {
    pub binary_to_data: Box<dyn Fn(&[bool]) -> Result<T, ()> + Send + Sync>,
    pub data_to_binary: Box<dyn Fn(T) -> Result<Vec<bool>, ()> + Send + Sync>,
}

pub struct DataLayer<T> {
    params: DataLayerParams<T>,
    network: Arc<RwLock<Network>>,
}

impl<T> DataLayer<T> {
    pub fn new(params: DataLayerParams<T>, network: Arc<RwLock<Network>>) -> Self {
        DataLayer { params, network }
    }

    pub fn get_network(&self) -> Arc<RwLock<Network>> {
        Arc::clone(&self.network)
    }

    pub fn replace_network(&mut self, network: Network) {
        let mut current_network = self.network.write().unwrap();
        *current_network = network;
    }

    pub fn push_data_binary_and_apply(&mut self, bit_vec: &[bool]) {
        self.network.write().unwrap().push_data_and_apply(bit_vec);
    }

    pub fn push_data_and_apply(&mut self, data: T) {
        let bit_vec_result = (self.params.data_to_binary)(data);

        if let Ok(bit_vec) = bit_vec_result {
            self.push_data_binary_and_apply(&bit_vec);
        }
    }

    pub fn push_data(&mut self, data: T) {
        let bit_vec_result = (self.params.data_to_binary)(data);

        if let Ok(bit_vec) = bit_vec_result {
            self.network.write().unwrap().push_data_binary(&bit_vec);
        }
    }

    pub fn predict_binary(&mut self, bit_vec: &[bool]) -> Vec<bool> {
        let binary_result = self.network.write().unwrap().predict(&bit_vec);

        binary_result
    }

    pub fn deserialize(&mut self, binary_result: &[bool]) -> Result<T, ()> {
        let data_result = (self.params.binary_to_data)(binary_result);

        data_result
    }

    pub fn predict(&mut self, data: T) -> Vec<bool> {
        let bit_vec = (self.params.data_to_binary)(data).unwrap();

       self.predict_binary(&bit_vec)
    }

    pub fn predict_and_deserialize(&mut self, data: T) -> Result<T, ()> {
        let binary_result = self.predict(data);

        self.deserialize(&binary_result)
    }
}
