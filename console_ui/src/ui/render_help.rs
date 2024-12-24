use ratatui::{
    layout::Rect,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

pub fn render_help(frame: &mut Frame, area: Rect) {
    let text = vec![
        Line::from(Span::raw("(h) to show help")),
        Line::from(Span::raw("(q) to quit")),
        Line::from(Span::raw("(Tab) to switch layer")),
        Line::from(Span::raw("(arrows) to select neuron")),
        Line::from(Span::raw("(n) to show states of neurons")),
        Line::from(Span::raw("(a) to show accumulated weights")),
        Line::from(Span::raw("(d) to show distance weights")),
        Line::from(Span::raw("(r) to show refract timeouts")),
        Line::from(Span::raw("(s) to save network state into file")),
        Line::from(Span::raw("(0) (1) to input signal into network")),
        Line::from(Span::raw("(Enter) to push signal and apply it immediately")),
        Line::from(Span::raw("(b) to push signal to network buffer")),
        Line::from(Span::raw("(t) to apply last action in network")),
    ];

    let paragraph = Paragraph::new(text)
        .block(Block::default().title("Help").borders(Borders::ALL))
        .style(Style::default());

    frame.render_widget(paragraph, area);
}
