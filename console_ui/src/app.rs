use chrono::Local;
use std::fs::File;
use std::io::Write;
use std::{sync::Arc, sync::RwLock};

use rnn_core::Network;

pub enum CurrentScreen {
    Help,
    Neurons,
    AccumulatedWeights,
    DistanceWeights,
    RefractTimeouts,
}

pub enum Layer {
    Layer1,
    Layer2,
}

pub struct App {
    pub buffer: String,
    pub current_screen: CurrentScreen,
    pub layer: Layer,
    pub neuron_x: usize,
    pub neuron_y: usize,
    network: Arc<RwLock<Network>>,
}

impl App {
    pub fn new(network: Arc<RwLock<Network>>) -> Self {
        App {
            buffer: String::new(),
            current_screen: CurrentScreen::Neurons,
            layer: Layer::Layer1,
            network,
            neuron_x: 0,
            neuron_y: 0,
        }
    }

    pub fn get_network(&self) -> Arc<RwLock<Network>> {
        Arc::clone(&self.network)
    }

    fn pop_buffer(&mut self) -> Vec<bool> {
        let mut res = Vec::<bool>::new();

        for c in self.buffer.chars() {
            match c {
                '0' => res.push(false),
                '.' => res.push(false),
                '-' => res.push(false),
                _ => res.push(true),
            }
        }

        self.buffer = String::new();

        res
    }

    pub fn push_data_and_apply(&mut self) {
        let data = self.pop_buffer();

        self.network.write().unwrap().push_data_and_apply(&data);
    }

    pub fn push_data_to_buffer(&mut self) {
        let data = self.pop_buffer();

        self.network.write().unwrap().push_data_binary(&data);
    }

    pub fn step(&mut self) {
        let mut network = self.network.write().unwrap();

        if network.has_next_action() {
            network.apply_next_action();
            return;
        }

        if network.has_signal_in_buffer() {
            network.read_from_buffer();
            network.apply_next_action();
        }
    }

    pub fn left(&mut self) {
        if self.neuron_x > 0 {
            self.neuron_x -= 1;
        }
    }

    pub fn right(&mut self) {
        let (row_width, _) = self.network.read().unwrap().get_layer_dimensions();

        if self.neuron_x < row_width - 1 {
            self.neuron_x += 1;
        }
    }

    pub fn up(&mut self) {
        if self.neuron_y > 0 {
            self.neuron_y -= 1;
        }
    }

    pub fn down(&mut self) {
        let (_, column_height) = self.network.read().unwrap().get_layer_dimensions();

        if self.neuron_y < column_height - 1 {
            self.neuron_y += 1;
        }
    }

    pub fn save_state(&self) -> std::io::Result<()> {
        let json_str = self.network.read().unwrap().get_json_dump();

        let now = Local::now();

        let filename = format!("dump_{}.json", now.format("%Y-%m-%d_%H-%M-%S"));

        let mut file = File::create(&filename)?;

        file.write_all(json_str.as_bytes())?;

        Ok(())
    }
}
