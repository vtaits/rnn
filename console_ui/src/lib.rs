mod app;
mod take_event;
mod ui;

use app::App;
use ratatui::{
    backend::{Backend, CrosstermBackend},
    crossterm::{
        event::{DisableMouseCapture, EnableMouseCapture},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    Terminal,
};
use rnn_core::Network;
use std::{error::Error, io};
use take_event::take_event;
use ui::ui;

pub fn run_console_app<T>(network: Box<Network<T>>) -> Result<(), Box<dyn Error>> {
    // setup terminal
    enable_raw_mode()?;
    let mut stderr = io::stderr(); // This is a special case. Normally using stdout is fine
    execute!(stderr, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stderr);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it
    let mut app = App::new(network);
    let _ = run_app(&mut terminal, &mut app);

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

fn run_app<B: Backend, T>(terminal: &mut Terminal<B>, app: &mut App<T>) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        let res = take_event(app);

        if let Ok(is_exit) = res {
            if is_exit {
                return Ok(());
            }
        }
    }
}
