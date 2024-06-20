use std::rc::Rc;

pub struct Dependency {
  weight: f32,
  target: Rc<Neuron>,
}

pub struct Neuron {
  dependencies: Vec<Dependency>,
  is_excited: bool,
}

impl Neuron {
  pub fn new() -> Neuron {
    return Neuron {
      dependencies: Vec::new(),
      is_excited: false,
    };
  }

  pub fn add_dependency(&mut self, other_neuron: &Rc<Neuron>, initial_weight: f32) {
    self.dependencies.push(Dependency {
      target: Rc::clone(other_neuron),
      weight: initial_weight,
    });
  }
}
