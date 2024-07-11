use crate::app::{App, CurrentScreen, Layer};
use ratatui::crossterm::event::{self, Event, KeyCode};
use std::io;

pub fn take_event(app: &mut App) -> io::Result<bool> {
    if let Event::Key(key) = event::read()? {
        if key.kind == event::KeyEventKind::Release {
            // Skip events that are not KeyEventKind::Press
            return Ok(false);
        }

        match key.code {
            KeyCode::Char('q') => {
                return Ok(true);
            }
            KeyCode::Char('n') => {
                app.current_screen = CurrentScreen::Neurons;
            }
            KeyCode::Char('a') => {
                app.current_screen = CurrentScreen::AccumulatedWeights;
            }
            KeyCode::Char('d') => {
                app.current_screen = CurrentScreen::DistanceWeights;
            }
            KeyCode::Backspace => {
                app.buffer.pop();
            }
            KeyCode::Char('+') | KeyCode::Char('1') => {
                app.buffer.push('+');
            }
            KeyCode::Char('.') | KeyCode::Char('0') | KeyCode::Char('-') => {
                app.buffer.push('-');
            }
            KeyCode::Enter => {
                app.tick_buffer();
            }
            KeyCode::Esc => {
                app.current_screen = CurrentScreen::Neurons;
            }
            KeyCode::Tab => {
                app.layer = match app.layer {
                    Layer::Layer1 => Layer::Layer2,
                    Layer::Layer2 => Layer::Layer1,
                };
            }
            KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right => {
                match key.code {
                    KeyCode::Up => {
                        app.up();
                    }
                    KeyCode::Down => {
                        app.down();
                    }
                    KeyCode::Left => {
                        app.left();
                    }
                    KeyCode::Right => {
                        app.right();
                    }
                    _ => {}
                };
            }
            _ => {}
        }
    }

    return Ok(false);
}
