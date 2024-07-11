use std::fmt::format;

use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Cell, Row, Table},
    Frame,
};

use crate::app::{App, Layer};

pub fn render_accumulated_weights(frame: &mut Frame, area: Rect, app: &App) {
    let network = app.get_network();
    let network_params = network.get_params();

    let layer_index = match app.layer {
        Layer::Layer1 => 1,
        Layer::Layer2 => 2,
    };

    let weights = network.get_neuron_accumulated_weights(layer_index, app.neuron_x, app.neuron_y);

    let mut widths: Vec<u16> = vec![];
    let mut full_width = 0;

    for _ in 0..network_params.layer_width {
        for _ in 0..network_params.field_width {
            widths.push(2);
            full_width += 2;
        }

        widths.push(1);
        full_width += 1;
    }

    let mut rows: Vec<Row> = Vec::new();

    let highlight_style = Style::default().bg(Color::Blue);

    for y in 0..weights.dim().1 {
        let mut cells: Vec<Cell> = Vec::new();

        for x in 0..weights.dim().0 {
            let value = weights[[x, y]];

            let cell_content = if value > 0.0 {
                format!("{}", value)
            } else {
                String::from("·")
            };

            let cell = if x == app.neuron_x && y == app.neuron_y {
                Cell::from(cell_content).style(highlight_style)
            } else {
                Cell::from(cell_content)
            };

            cells.push(cell);

            if (x + 1) % network_params.field_width == 0 {
                cells.push(Cell::from(String::from("|")));
            }
        }

        rows.push(Row::new(cells));

        if (y + 1) % network_params.field_height == 0 {
            let mut dash_cells: Vec<String> = Vec::new();
            for width in widths.iter() {
                let cell_content = "-".repeat(*width as usize);

                dash_cells.push(cell_content);
            }

            rows.push(Row::new(dash_cells));
        }
    }

    let table: Table = Table::new(rows, widths)
        .block(
            Block::default()
                .title("Accumulated weights")
                .borders(Borders::ALL),
        )
        .style(Style::default().fg(Color::White));

    frame.render_widget(table, area);
}
