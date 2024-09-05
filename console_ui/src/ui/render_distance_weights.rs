use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Cell, Row, Table},
    Frame,
};

use crate::app::{App, Layer};

pub fn render_distance_weights(frame: &mut Frame, area: Rect, app: &App) {
    let network_rc = app.get_network();
    let network_ref = network_rc.read().unwrap();
    let layer_params = network_ref.get_layer_params();

    let layer_index = match app.layer {
        Layer::Layer1 => 1,
        Layer::Layer2 => 2,
    };

    let weights = network_ref.get_neuron_distance_weights(layer_index, app.neuron_x, app.neuron_y);

    let mut widths: Vec<u16> = vec![];

    for _ in 0..layer_params.layer_width {
        for _ in 0..layer_params.field_width {
            widths.push(5);
        }

        widths.push(1);
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

            if (x + 1) % layer_params.field_width == 0 {
                cells.push(Cell::from(String::from("|")));
            }
        }

        rows.push(Row::new(cells));

        if (y + 1) % layer_params.field_height == 0 {
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
                .title("Distance weights")
                .borders(Borders::ALL),
        )
        .style(Style::default());

    frame.render_widget(table, area);
}
