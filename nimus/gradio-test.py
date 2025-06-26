import gradio as gr

def recommender():
    with gr.Blocks():
        gr.Markdown("## 🧠 Recommender Dashboard with Images")
        with gr.Row():
            min_input = gr.Number(label="Min Past Interactions", value=5, precision=1)
            max_input = gr.Number(label="Max Past Interactions (zero for no limit)", value=0, precision=1)
        with gr.Row():
            category_dropdown = gr.Dropdown(label="Main Category Filter (Any for no filtering - much faster)",
                                            choices=["A", "B", "C", "Any"], value="Any")
            category_min_count = gr.Number(label="Min Interactions With Category", value=1, precision=1)
            run_button = gr.Button("Pick Random User")

        gr.Markdown("### 20/8001 users matching search criteria")
        with gr.Row():
            left_btn = gr.Button("<")
            current_user_no = gr.Number(label="Current user", value=1, precision=1)
            right_btn = gr.Button(">")

        user_out = gr.Textbox(label="User ID", interactive=False)
        with gr.Column():
            gr.Markdown("### 🔁 Past Interactions")
            hist_gallery = gr.Gallery(label="History", columns=5, height="auto")
        with gr.Column():
            gr.Markdown("### 🌟 Recommendations")
            rec_gallery = gr.Gallery(label="Recommendations", columns=5, height="auto")

def user_browser():
    with gr.Blocks():
        gr.Markdown("## User Interactions")
        user_out = gr.Textbox(label="User ID", interactive=False)
        with gr.Column():
            gr.Markdown("### 🔁 Past Interactions")
            hist_gallery = gr.Gallery(label="History", columns=5, height="auto")

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Team 6 Recommender App")
        with gr.Tab("Recommender"):
            recommender()
        with gr.Tab("User Browser"):
            user_browser()

    demo.launch()

if __name__ == "__main__":
    main()