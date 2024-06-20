use crate::neuron::Neuron;
use std::{borrow::BorrowMut, rc::Rc};

/// Parameters of the layer
pub struct LayerParams {
  /// Width in neurons of one field
  field_width: usize,
  /// Height in neurons of one field
  field_height: usize,
  /// Width in fields of one layer
  layer_width: usize,
  // Height in fields of one layer
  layer_height: usize,
}

pub struct Layer<'a> {
  neurons: Vec<Rc<Neuron>>,
  params: &'a LayerParams,
}

impl <'a> Layer<'a> {
  pub fn new(params: &'a LayerParams) -> Layer<'a> {
    let capacity = params.field_width * params.field_height * params.layer_width * params.layer_height;

    let neurons: Vec<Rc<Neuron>> = Vec::with_capacity(capacity);

    return Layer {
      neurons,
      params,
    };
  }

  pub fn connect(&mut self, other_layer: &'a mut Layer) {
    let o = other_layer.neurons.get_mut(0).unwrap();
    
    for (self_index, self_neuron) in self.neurons.iter_mut().enumerate() {
      self_neuron.borrow_mut().add_dependency(o, 0.0);
      /* for (other_index, other_neuron) in other_layer.neurons.iter_mut().enumerate() {
        self_neuron.add_dependency(other_neuron, if self_index == other_index { 1.0 } else { 0.0 });
        other_neuron.add_dependency(self_neuron, if self_index == other_index { 1.0 } else { 0.0 });
      } */
    }
  }
}
