mod app;
mod take_event;
mod ui;

use std::{error::Error, io};
use app::App;
use ratatui::{
  backend::{Backend, CrosstermBackend},
  crossterm::{
      event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
      execute,
      terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
  },
  Terminal,
};
use rnn_core::Network;
use take_event::take_event;

fn run<'a>(network: &'a mut Network) -> Result<(), Box<dyn Error>> {
  // setup terminal
  enable_raw_mode()?;
  let mut stderr = io::stderr(); // This is a special case. Normally using stdout is fine
  execute!(stderr, EnterAlternateScreen, EnableMouseCapture)?;
  let backend = CrosstermBackend::new(stderr);
  let mut terminal = Terminal::new(backend)?;

  // create app and run it
  let mut app = App::new(network);
  run_app(&mut terminal, &mut app);

  // restore terminal
  disable_raw_mode()?;
  execute!(
      terminal.backend_mut(),
      LeaveAlternateScreen,
      DisableMouseCapture
  )?;
  terminal.show_cursor()?;

  Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) {
  loop {
      terminal.draw(|f| ui(f, app))?;

      let res = take_event(app);

      if let Ok(is_exit) = res {
        if is_exit {
          return;
        }
      }
  }
}
