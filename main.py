import streamlit as st

st.set_page_config(page_title="streamlineML", page_icon="favicon.png", layout="wide")

home = st.Page("app.py", title="Home")
about = st.Page("screens/about.py", title="About", icon=":material/info:")
terms = st.Page("screens/terms.py", title="Terms of Use", icon=":material/hammer:")
privacy = st.Page("screens/privacy.py", title="Privacy Policy", icon=":material/prescription:")

router = st.navigation([home, about, terms, privacy])

if __name__ == "__main__":
    router.run()
