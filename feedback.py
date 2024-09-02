import streamlit as st
import re

# path to the csv file
csv_file_path = "app_feedback.csv"

# Save feedback to the csv file
def save_feedback(name, email, feedback, rating):
    with open(csv_file_path, "a") as file:
        file.write(f"{name}, {email}, {feedback}, {rating}\n")

# Validate email format
def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.com$'
    if re.match(pattern, email):
        return True
    return False
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

def main():
    feedback_form()

if __name__ == "__main__":
    main()