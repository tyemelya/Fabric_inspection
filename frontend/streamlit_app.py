import streamlit as st
import requests

API_URL = st.secrets["API_URL"]

st.title("Fabric Inspection Demo")

uploaded_file = st.file_uploader(
    "Upload fabric image",
    type=["jpg", "jpeg", "png"]
)
question = st.text_input(
    "Inspection question",
    "What defect is visible?"
)

st.caption(
    """
    You can ask your own question about the fabric image.

    Possible examples:

    - What defect is visible?
    - Find similar examples
    - Why was this classified as a broken stitch?
    - Compare this with previous cases
    """
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", width=300)

if st.button("Start inspection"):
    if uploaded_file is None:
        st.warning("Upload an image first.")
    else:
        files = {
            "uploaded_image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }
        data = {
            "question": question
        }

        with st.spinner("Inspecting..."):
            response = requests.post(
                f"{API_URL}/inspect",
                files=files,
                data=data
            )

        if response.status_code == 200:
            result = response.json()

            st.subheader("Report")
            st.write(result["report"])
            st.subheader("Prediction")
            st.write(
                f"""
                Defect: {result["prediction"]}

                Confidence:
                {result["vision_confidence"]:.2%}
                """
            )

            if result["retrieved_cases"]:
                st.subheader("Similar cases")
                cases = result["retrieved_cases"]
                cols = st.columns(len(cases))

                for col, case in zip(cols, cases):
                    with col:
                        image_response = requests.get(
                            f"{API_URL}/images/{case['image_id']}"
                        )
                        if image_response.status_code == 200:
                            st.image(
                                image_response.content,
                                caption=(
                                    f"ID: {case['image_id']}\n"
                                    f"{case['defect_class']}\n"
                                    f"Distance: {case['distance']:.3f}"
                                ),
                                use_container_width=True
                            )
                        else:
                            st.error(
                                f"Cannot load image {case['image_id']}"
                            )
        else:
            st.error(response.text)