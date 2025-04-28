import streamlit as st
import time
# TODO: WORK ON THIS
st.title("streamlineML")
st.info("Somthing cool is coming soon!")

def main():
    if int(time.time()) % 2 == 0:
        st.balloons()

while True:
    main()
    time.sleep(1)
    st.rerun()