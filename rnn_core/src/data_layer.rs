use std::{cell::RefCell, rc::Rc};

use crate::Network;

pub struct DataLayerParams<T> {
    pub binary_to_data: Box<dyn Fn(Vec<bool>) -> Result<T, ()>>,
    pub data_to_binary: Box<dyn Fn(T) -> Result<Vec<bool>, ()>>,
}

pub struct DataLayer<T> {
    params: DataLayerParams<T>,
    network: Rc<RefCell<Network>>,
}

impl<T> DataLayer<T> {
    pub fn new(params: DataLayerParams<T>, network: Rc<RefCell<Network>>) -> Self {
        DataLayer { params, network }
    }

    pub fn get_network(&self) -> Rc<RefCell<Network>> {
        Rc::clone(&self.network)
    }

    pub fn replace_network(&mut self, network: Rc<RefCell<Network>>) {
        self.network = network;
    }

    pub fn push_data(&mut self, data: T) {
        let bit_vec_result = (self.params.data_to_binary)(data);

        if let Ok(bit_vec) = bit_vec_result {
            self.network.borrow_mut().push_data_binary(&bit_vec);
        }
    }
}
