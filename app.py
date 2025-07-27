import joblib
import numpy as np
import pandas as pd
import gradio as gr
import altair as alt

# Load the trained model
pipe_lr = joblib.load(open("model/text_emotion (1).pkl", "rb"))

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}


# Emotion prediction function
def predict_emotion(text):
    prediction = pipe_lr.predict([text])[0]
    probability = pipe_lr.predict_proba([text])[0]

    # Prepare chart data
    proba_df = pd.DataFrame({
        'emotions': pipe_lr.classes_,
        'probability': probability
    })

    chart = alt.Chart(proba_df).mark_bar().encode(
        x='emotions',
        y='probability',
        color='emotions'
    ).properties(title='Prediction Probabilities')

    return (
        f"**Prediction:** {prediction} {emotions_emoji_dict.get(prediction, '')}\n\n"
        f"**Confidence:** {np.max(probability):.2f}",
        proba_df,
        chart
    )


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Text Emotion Detection")

    with gr.Row():
        text_input = gr.Textbox(label="Enter Text", placeholder="Type something...", lines=4)
        predict_btn = gr.Button("Predict Emotion")

    with gr.Row():
        output_label = gr.Markdown()

    with gr.Row():
        proba_table = gr.Dataframe(headers=["Emotion", "Probability"], label="Probability Table")
        chart_output = gr.Plot(label="Probability Bar Chart")

    predict_btn.click(
        fn=predict_emotion,
        inputs=text_input,
        outputs=[output_label, proba_table, chart_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)

