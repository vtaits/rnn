use std::{sync::Arc, sync::RwLock};

use crate::Network;

pub struct DataLayerParams<T> {
    pub binary_to_data: Box<dyn Fn(Vec<bool>) -> Result<T, ()> + Send + Sync>,
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

    pub fn replace_network(&mut self, network: Arc<RwLock<Network>>) {
        self.network = network;
    }

    pub fn push_data(&mut self, data: T) {
        let bit_vec_result = (self.params.data_to_binary)(data);

        if let Ok(bit_vec) = bit_vec_result {
            self.network.write().unwrap().push_data_binary(&bit_vec);
        }
    }

    pub fn predict(&mut self, data: T) -> Result<T, ()> {
        let bit_vec = (self.params.data_to_binary)(data)?;

        let binary_result = self.network.write().unwrap().predict(&bit_vec);

        let data_result = (self.params.binary_to_data)(binary_result);

        data_result
    }
}
