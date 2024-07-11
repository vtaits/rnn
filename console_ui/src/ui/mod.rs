mod render_neurons;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, CurrentScreen, Layer};

use render_neurons::render_neurons;

fn get_title<'a>(app: &'a App) -> Paragraph<'a> {
    let text = vec![
        match app.layer {
            Layer::Layer1 => Span::styled("Layer 1", Style::default()),
            Layer::Layer2 => Span::styled("Layer 2", Style::default()),
        }
        .fg(Color::LightGreen)
        .to_owned(),
        Span::styled(" | ", Style::default().fg(Color::White)),
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
        Span::styled(" | ", Style::default().fg(Color::White)),
        Span::styled("Input: ", Style::default().fg(Color::Gray)),
        Span::styled(&app.buffer, Style::default().fg(Color::White)),
    ];

    return Paragraph::new(Line::from(text)).block(Block::default().borders(Borders::ALL));
}

fn get_keys_hint<'a>(app: &App) -> Paragraph<'a> {
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

pub fn ui(f: &mut Frame, app: &App) {
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

    render_neurons(f, chunks[1], app);

    let keys_hint = get_keys_hint(app);

    f.render_widget(keys_hint, chunks[2]);

    /*  let list = List::new(list_items);

    f.render_widget(list, chunks[1]);
    let current_navigation_text = vec![
        // The first half of the text
        match app.current_screen {
            CurrentScreen::Main => Span::styled("Normal Mode", Style::default().fg(Color::Green)),
            CurrentScreen::Editing => {
                Span::styled("Editing Mode", Style::default().fg(Color::Yellow))
            }
            CurrentScreen::Exiting => Span::styled("Exiting", Style::default().fg(Color::LightRed)),
        }
        .to_owned(),
        // A white divider bar to separate the two sections
        Span::styled(" | ", Style::default().fg(Color::White)),
        // The final section of the text, with hints on what the user is editing
        {
            if let Some(editing) = &app.currently_editing {
                match editing {
                    CurrentlyEditing::Key => {
                        Span::styled("Editing Json Key", Style::default().fg(Color::Green))
                    }
                    CurrentlyEditing::Value => {
                        Span::styled("Editing Json Value", Style::default().fg(Color::LightGreen))
                    }
                }
            } else {
                Span::styled("Not Editing Anything", Style::default().fg(Color::DarkGray))
            }
        },
    ];

    let mode_footer = Paragraph::new(Line::from(current_navigation_text))
        .block(Block::default().borders(Borders::ALL));

    let current_keys_hint = {
        match app.current_screen {
            CurrentScreen::Main => Span::styled(
                "(q) to quit / (e) to make new pair",
                Style::default().fg(Color::Red),
            ),
            CurrentScreen::Editing => Span::styled(
                "(ESC) to cancel/(Tab) to switch boxes/enter to complete",
                Style::default().fg(Color::Red),
            ),
            CurrentScreen::Exiting => Span::styled(
                "(q) to quit / (e) to make new pair",
                Style::default().fg(Color::Red),
            ),
        }
    };

    let key_notes_footer =
        Paragraph::new(Line::from(current_keys_hint)).block(Block::default().borders(Borders::ALL));

    let footer_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[2]);

    f.render_widget(mode_footer, footer_chunks[0]);
    f.render_widget(key_notes_footer, footer_chunks[1]);

    if let Some(editing) = &app.currently_editing {
        let popup_block = Block::default()
            .title("Enter a new key-value pair")
            .borders(Borders::NONE)
            .style(Style::default().bg(Color::DarkGray));

        let area = centered_rect(60, 25, f.size());
        f.render_widget(popup_block, area);

        let popup_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .margin(1)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        let mut key_block = Block::default().title("Key").borders(Borders::ALL);
        let mut value_block = Block::default().title("Value").borders(Borders::ALL);

        let active_style = Style::default().bg(Color::LightYellow).fg(Color::Black);

        match editing {
            CurrentlyEditing::Key => key_block = key_block.style(active_style),
            CurrentlyEditing::Value => value_block = value_block.style(active_style),
        };

        let key_text = Paragraph::new(app.key_input.clone()).block(key_block);
        f.render_widget(key_text, popup_chunks[0]);

        let value_text = Paragraph::new(app.value_input.clone()).block(value_block);
        f.render_widget(value_text, popup_chunks[1]);
    }

    if let CurrentScreen::Exiting = app.current_screen {
        f.render_widget(Clear, f.size()); //this clears the entire screen and anything already drawn
        let popup_block = Block::default()
            .title("Y/N")
            .borders(Borders::NONE)
            .style(Style::default().bg(Color::DarkGray));

        let exit_text = Text::styled(
            "Would you like to output the buffer as json? (y/n)",
            Style::default().fg(Color::Red),
        );
        // the `trim: false` will stop the text from being cut off when over the edge of the block
        let exit_paragraph = Paragraph::new(exit_text)
            .block(popup_block)
            .wrap(Wrap { trim: false });

        let area = centered_rect(60, 25, f.size());
        f.render_widget(exit_paragraph, area);
    } */
}

/// helper function to create a centered rect using up certain percentage of the available rect `r`
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    // Cut the given rectangle into three vertical pieces
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    // Then cut the middle vertical piece into three width-wise pieces
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1] // Return the middle chunk
}
