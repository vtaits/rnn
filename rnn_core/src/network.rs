use crate::layer::{Layer, LayerParams};

pub struct Network<'a> {
  top_layer: Layer<'a>,
  bottom_layer: Layer<'a>,
}

impl <'a> Network<'a> {
  pub fn new(params: &'a LayerParams) -> Network<'a> {
    let top_layer = Layer::new(params);
    let bottom_layer = Layer::new(params);

    return Network {
      top_layer,
      bottom_layer,
    };
  }
}