import streamlit as st
from Modules.home import home_page

def main():
    """
    Main function to manage the Streamlit app.
    """
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home"])

    if page == "Home":
        home_page()

if __name__ == "__main__":
    main()