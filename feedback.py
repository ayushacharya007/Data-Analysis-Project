import streamlit as st
import sqlite3
import re

# Connect to the existing database
def connect_db():
    return sqlite3.connect('app_feedback')

# Save feedback to the database
def save_feedback(name, email, feedback, rating):
    conn = connect_db()
    c = conn.cursor()
    c.execute('''
        INSERT INTO feedback (name, email, feedback, rating)
        VALUES (?, ?, ?, ?)
    ''', (name, email, feedback, rating))
    conn.commit()
    conn.close()

# Validate email format
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.com$'
    return re.match(pattern, email) is not None

# Create a feedback form
def feedback_form():
    st.write("*Feedback Form*")
    st.info("Please fill in the required fields marked with *")
    name = st.text_input(":blue[Name] :red[*] :")
    email = st.text_input(":blue[Email] :")
    feedback = st.text_input(":blue[Feedback]  :red[*] :")
    rating = st.number_input(":blue[Rating] :red[*] :", min_value=1, max_value=5)

    submit = st.button("Submit")    
    if submit:
        if not name:
            st.error('Name cannot be empty')
        elif not feedback:
            st.error('Feedback cannot be empty')
        elif email and not is_valid_email(email):
            st.error('Invalid email format')
        else:
            save_feedback(name, email, feedback, rating)
            st.success('Feedback submitted successfully')
    return name, email, feedback, rating


# Main function to run the Streamlit app
def main():
    feedback_form()

if __name__ == "__main__":
    main()