mod render_accumulated_weights;
mod render_distance_weights;
mod render_neurons;

use ratatui::{
    layout::{Constraint, Direction, Layout},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use render_distance_weights::render_distance_weights;

use crate::app::{App, CurrentScreen, Layer};

use render_accumulated_weights::render_accumulated_weights;
use render_neurons::render_neurons;

fn get_title<T>(app: &App<T>) -> Paragraph {
    let text = vec![
        match app.layer {
            Layer::Layer1 => Span::styled("Layer 1", Style::default()),
            Layer::Layer2 => Span::styled("Layer 2", Style::default()),
        }
        .fg(Color::LightGreen)
        .to_owned(),
        Span::styled(" | ", Style::default()),
        match app.current_screen {
            CurrentScreen::Neurons => Span::styled("Neurons", Style::default()),
            CurrentScreen::AccumulatedWeights => Span::styled(
                format!("Accumulated for {}, {}", app.neuron_x, app.neuron_y),
                Style::default(),
            ),
            CurrentScreen::DistanceWeights => Span::styled(
                format!("Distances for {}, {}", app.neuron_x, app.neuron_y),
                Style::default(),
            ),
        }
        .to_owned(),
        Span::styled(" | ", Style::default()),
        Span::styled("Input: ", Style::default().fg(Color::Gray)),
        Span::styled(&app.buffer, Style::default()),
    ];

    return Paragraph::new(Line::from(text)).block(Block::default().borders(Borders::ALL));
}

fn get_keys_hint<T>(app: &App<T>) -> Paragraph {
    let current_keys_hint = {
        match app.current_screen {
              CurrentScreen::Neurons => Span::styled(
                  "(Tab) to switch layer / (q) to quit / arrows to select neuron / (a) to show accumulated weights for neuron / (d) to show distance weights for neuron",
                  Style::default().fg(Color::Red),
              ),
              CurrentScreen::AccumulatedWeights => Span::styled(
                  "(Tab) to switch layer / (q) to quit / (n) to state of the layer / (d) to show distance weights for neuron",
                  Style::default().fg(Color::Red),
              ),
              CurrentScreen::DistanceWeights => Span::styled(
                  "(Tab) to switch layer / (q) to quit / (n) to state of the layer / (a) to show accumulated weights for neuron",
                  Style::default().fg(Color::Red),
              ),
          }
    };

    return Paragraph::new(Line::from(current_keys_hint))
        .block(Block::default().borders(Borders::ALL));
}

pub fn ui<T>(f: &mut Frame, app: &mut App<T>) {
    // Create the layout sections.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(f.size());

    let title = get_title(app);

    f.render_widget(title, chunks[0]);

    match app.current_screen {
        CurrentScreen::AccumulatedWeights => {
            render_accumulated_weights(f, chunks[1], app);
        }
        CurrentScreen::DistanceWeights => {
            render_distance_weights(f, chunks[1], app);
        }
        CurrentScreen::Neurons => {
            render_neurons(f, chunks[1], app);
        }
    }

    let keys_hint = get_keys_hint(app);

    f.render_widget(keys_hint, chunks[2]);
}
