from pathlib import Path
import html
import ipywidgets as widgets
from IPython.display import display, Markdown, clear_output


img = widgets.Image(format="png")

img.layout = widgets.Layout(
    width="100%",
    max_width="700px",
    height="auto",
    border="1px solid #ddd",
    padding="8px",
)


def show_widget(result):
    agent_dropdown = widgets.Dropdown(
        options=list(result["agent_findings"].keys()),
        value=list(result["agent_findings"].keys())[0],
        description="🤖 Agent:",
        layout=widgets.Layout(width="320px"),
    )

    visual_dropdown = widgets.Dropdown(
        options=[],
        description="🔗 Reference:",
        layout=widgets.Layout(width="460px"),
    )

    toggle_report_btn = widgets.Button(
        description="📄 Final Report",
        button_style="info",
        icon="file-alt",
        layout=widgets.Layout(width="170px"),
    )

    path_label = widgets.HTML(
        layout=widgets.Layout(
            width="100%",
            max_width="460px",
        )
    )

    right_out = widgets.Output(
        layout=widgets.Layout(
            width="100%",
            height="650px",
            overflow_y="auto",
            border="1px solid #ddd",
            padding="12px",
        )
    )

    showing_final_report = False

    # -----------------------------
    # Helpers
    # -----------------------------

    def show_image_path(image_path):
        if not image_path:
            img.value = b""
            path_label.value = "<div style='color:#666;'>No image selected</div>"
            return

        img.value = Path(image_path).read_bytes()

        filename = Path(image_path).name

        path_label.value = f"""
        <div style="
            color:#777;
            font-size:13px;
            margin-top:8px;
        ">
            📄 {html.escape(filename)}
        </div>
        """

    def render_right_column():
        with right_out:
            clear_output(wait=True)

            if showing_final_report:
                display(Markdown(result.get("final_report", "_No final report._")))
                return

            agent_name = agent_dropdown.value
            agent_findings = result["agent_findings"].get(agent_name, [])

            if agent_findings:
                display(Markdown("\n\n".join(agent_findings)))
            else:
                display(Markdown("_No findings for this agent._"))

    def update_visual_options():
        if showing_final_report:
            ref_to_image = result.get("final_self_ref_image_paths", {})
        else:
            agent_name = agent_dropdown.value
            ref_to_image = result["agent_self_ref_image_paths"].get(agent_name, {})

        visual_dropdown.description = "🔗 Reference:"
        visual_dropdown.options = [(ref, ref) for ref in ref_to_image]

        if ref_to_image:
            first_ref = next(iter(ref_to_image))
            visual_dropdown.value = first_ref
            show_image_path(ref_to_image[first_ref])
        else:
            visual_dropdown.value = None
            show_image_path(None)

    # -----------------------------
    # Callbacks
    # -----------------------------

    def on_agent_change(change):
        if showing_final_report:
            return

        update_visual_options()
        render_right_column()

    def on_visual_change(change):
        ref = change["new"]

        if ref is None:
            return

        if showing_final_report:
            mapping = result.get("final_self_ref_image_paths", {})
        else:
            mapping = result["agent_self_ref_image_paths"].get(
                agent_dropdown.value,
                {},
            )

        show_image_path(mapping.get(ref))

    def on_toggle_report(_):
        nonlocal showing_final_report

        showing_final_report = not showing_final_report

        if showing_final_report:
            toggle_report_btn.description = "📝 Findings"
            toggle_report_btn.button_style = "warning"
            agent_dropdown.disabled = True
        else:
            toggle_report_btn.description = "📄 Final Report"
            toggle_report_btn.button_style = "info"
            agent_dropdown.disabled = False

        update_visual_options()
        render_right_column()

    agent_dropdown.observe(on_agent_change, names="value")
    visual_dropdown.observe(on_visual_change, names="value")
    toggle_report_btn.on_click(on_toggle_report)

    # -----------------------------
    # Layout
    # -----------------------------

    top_controls = widgets.HBox(
        [
            agent_dropdown,
            visual_dropdown,
            widgets.Box(layout=widgets.Layout(flex="1")),
            toggle_report_btn,
        ],
        layout=widgets.Layout(
            justify_content="flex-start",
            align_items="center",
            gap="16px",
            width="100%",
            margin="0 0 14px 0",
        ),
    )

    left_col = widgets.VBox(
        [
            widgets.HTML("<h3 style='margin:0 0 12px 0;'>🖼 Visual Grounding</h3>"),
            img,
            path_label,
        ],
        layout=widgets.Layout(
            width="50%",
            border="1px solid #ddd",
            padding="16px",
            border_radius="8px",
        ),
    )

    right_col = widgets.VBox(
        [
            widgets.HTML(
                "<h3 style='margin:0 0 12px 0;'>📝 Findings / Final Report</h3>"
            ),
            right_out,
        ],
        layout=widgets.Layout(
            width="50%",
            border="1px solid #ddd",
            padding="16px",
            border_radius="8px",
        ),
    )

    content = widgets.HBox(
        [left_col, right_col],
        layout=widgets.Layout(
            justify_content="flex-start",
            align_items="flex-start",
            gap="32px",
            width="100%",
            max_width="none",
        ),
    )

    title = widgets.HTML("""
    <h2 style="
        margin:0 0 16px 0;
        font-weight:600;
    ">
    📊 Annual Report Explorer
    </h2>
    """)

    viewer = widgets.VBox(
        [title, top_controls, content],
        layout=widgets.Layout(
            align_items="stretch",
            width="100%",
            padding="12px 0",
        ),
    )

    # -----------------------------
    # Initialize
    # -----------------------------

    update_visual_options()
    render_right_column()

    display(viewer)
