use ratatui::{
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Borders, Cell, Row, Table},
    Frame,
};

use crate::app::{App, Layer};

pub fn render_refract_intervals(frame: &mut Frame, area: Rect, app: &App) {
    let network_rc = app.get_network();
    let network_ref = network_rc.read().unwrap();
    let layer_params = network_ref.get_layer_params();

    let layer_index = match app.layer {
        Layer::Layer1 => 1,
        Layer::Layer2 => 2,
    };

    let mut widths = vec![];

    for _ in 0..layer_params.layer_width {
        for _ in 0..layer_params.field_width {
            widths.push(1);
        }

        widths.push(1);
    }

    let mut rows: Vec<Row> = Vec::new();

    let highlight_style = Style::default().bg(Color::Blue);

    let (target_layer_x, target_layer_y, target_neuron_in_field_x, target_neuron_in_field_y) =
        network_ref.get_neuron_full_coordinates(app.neuron_x, app.neuron_y);

    for layer_y in 0..layer_params.layer_height {
        for neuron_in_field_y in 0..layer_params.field_height {
            let mut cells: Vec<Cell> = Vec::new();

            for layer_x in 0..layer_params.layer_width {
                for neuron_in_field_x in 0..layer_params.field_width {
                    let refract_timeout = network_ref.get_neuron_refract_timeout(
                        layer_index,
                        layer_x,
                        layer_y,
                        neuron_in_field_x,
                        neuron_in_field_y,
                    );

                    let cell_content = if refract_timeout > 0 {
                        format!("{}", refract_timeout)
                    } else {
                        String::from("·")
                    };
                    let cell = if layer_x == target_layer_x
                        && layer_y == target_layer_y
                        && neuron_in_field_x == target_neuron_in_field_x
                        && neuron_in_field_y == target_neuron_in_field_y
                    {
                        Cell::from(cell_content).style(highlight_style)
                    } else {
                        Cell::from(cell_content)
                    };

                    cells.push(cell);
                }

                cells.push(Cell::from(""));
            }

            rows.push(Row::new(cells));
        }

        rows.push(Row::new(Vec::<String>::new()));
    }

    let table: Table = Table::new(rows, widths)
        .block(
            Block::default()
                .title("Refract timeouts")
                .borders(Borders::ALL),
        )
        .style(Style::default());

    frame.render_widget(table, area);
}
