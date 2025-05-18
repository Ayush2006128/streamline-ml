import streamlit as st

with open("../TERMS_OF_USE.md", "r") as file:
    terms_of_use = file.read()
    st.write(terms_of_use)