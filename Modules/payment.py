# modules/payment.py

import streamlit as st

def payment_options():
    st.write("### Payment Options")
    st.write("Choose your preferred payment method.")

    payment_methods = ["Bank Card", "Cash", "MT Cash", "Barid Pay"]
    method = st.selectbox("Select a payment method", payment_methods)

    st.write(f"You have selected: {method}")
    if st.button("Confirm Payment"):
        st.write("Payment successful!")
