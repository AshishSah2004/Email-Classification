import streamlit as st

st.title("Email Spam Detection")

email = st.text_area("Enter Email")

if st.button("Check"):

    st.write(email)
