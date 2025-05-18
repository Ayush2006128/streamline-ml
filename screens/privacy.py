import streamlit as st

with open("../PRIVACY_POLICY.md", "r") as file:
    privacy_policy = file.read()
    st.write(privacy_policy)