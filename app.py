import streamlit as st
from clarifai.client.model import Model
import base64
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()
import os

clarifai_pat = os.getenv("CLARIFAI_PAT")
openai_api_key = os.getenv("OPEN_AI")

def generate_image(user_description, api_key):
    prompt = f"Become a group of experts consisting of a child psychologist working every day with children in preschool, a preschool group educator, each of you with 15 years of experience. You are the best in the market, meaning that while others are rated 10 points for their work, you receive 100 points at the same time. For completing the task to the best of your ability, you will receive a substantial amount of money. Your task is to prepare a coloring page based on the user's description, meaning the image should not contain any colors, only the outlines of drawings on a theme specified by the user, the material must be suitable for the age of the children, i.e., 3-5 years old: {user_description}"
    inference_params = dict(quality="standard", size="1024x1024")
    model_prediction = Model(
        f"https://clarifai.com/openai/dall-e/models/dall-e-3?api_key={api_key}"
    ).predict_by_bytes(
        prompt.encode(), input_type="text", inference_params=inference_params
    )
    output_base64 = model_prediction.outputs[0].data.image.base64
    with open("generated_image.png", "wb") as f:
        f.write(output_base64)
    return "generated_image.png"


def main():
    
    st.set_page_config(page_title="Colorful Minds App", layout="wide")
    st.title("Colorful Minds App")

    with st.sidebar:
        st.header("Hi, I will prepare a coloring page for preschool children.")
        image_description = st.text_area("Provide the topic of the lesson.", height=50)
        generate_image_btn = st.button("Prepare material")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Your materials for classes with children")
        if generate_image_btn and image_description:
            with st.spinner("Generating for You..."):
                image_path = generate_image(image_description, clarifai_pat)
                if image_path:
                    st.image(
                        image_path,
                        caption="Your material for children.",
                        use_column_width=True,
                    )
                    st.success("Ready.")
                else:
                    st.error("Failed.")

    with col2:
        st.header("Another materials")
        if generate_image_btn and image_description:
            with st.spinner("Generating for You..."):
                image_path = generate_image(image_description, clarifai_pat)
                if image_path:
                    st.image(
                        image_path,
                        caption="Your material for children.",
                        use_column_width=True,
                    )
                    st.success("Ready.")
                else:
                    st.error("Failed.")


if __name__ == "__main__":
    main()
