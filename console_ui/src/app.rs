use rnn_core::{Network, NetworkParams};

pub enum CurrentScreen {
    Neurons,
    AccumulatedWeights,
    DistanceWeights,
}

pub enum Layer {
    Layer1,
    Layer2,
}

pub struct App<'a> {
    pub buffer: String,
    pub current_screen: CurrentScreen,
    pub layer: Layer,
    pub neuron_x: usize,
    pub neuron_y: usize,
    network: &'a mut Network<'a>,
}

impl<'a> App<'a> {
    pub fn new(network: &'a mut Network<'a>) -> App<'a> {
        App {
            buffer: String::new(),
            current_screen: CurrentScreen::Neurons,
            layer: Layer::Layer1,
            network,
            neuron_x: 0,
            neuron_y: 0,
        }
    }

    pub fn get_network(&self) -> &Network {
        return self.network;
    }

    pub fn get_network_params(&self) -> &NetworkParams {
        return self.network.get_params();
    }

    pub fn tick_buffer(&mut self) {
        let mut data = Vec::<bool>::new();

        for c in self.buffer.chars() {
            match c {
                '0' => data.push(false),
                '.' => data.push(false),
                '-' => data.push(false),
                _ => data.push(true),
            }
        }

        self.buffer = String::new();

        self.network.tick(&data);
    }

    pub fn left(&mut self) {
        if self.neuron_x > 0 {
            self.neuron_x -= 1;
        }
    }

    pub fn right(&mut self) {
        let (row_width, _) = self.network.get_layer_dimensions();

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
        let (_, column_height) = self.network.get_layer_dimensions();

        if self.neuron_y < column_height - 1 {
            self.neuron_y += 1;
        }
    }
}
