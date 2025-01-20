import requests
import streamlit as st


def fetch_image_from_api(api_url):
    response = requests.get(api_url)
    response.raise_for_status()
    return response.content


st.title("Pokémon Generator")
st.write("Click to generate a random novel Pokémon image.")

if st.button("Generate"):
    try:
        # Replace 'https://example.com/image-api' with your actual image API endpoint
        api_url = "https://picsum.photos/500"  # Example API for a random image
        image = fetch_image_from_api(api_url)

        # Display the image
        st.image(image, caption="Fetched Image", use_column_width=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch image: {e}")
