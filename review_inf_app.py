import pandas as pd
import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError

# Configure S3 client with credentials from Streamlit secrets
s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["aws_access_key_id"],
    aws_secret_access_key=st.secrets["aws_secret_access_key"],
    region_name=st.secrets["aws_region"]
)


def upload_file():
    """Handles file upload and checks if required columns exist."""
    # uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    bucket = "derm-data-general"
    key = "dev_set_2.csv"
    try:
        uploaded_file = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        # return response
    except NoCredentialsError:
        st.error("No AWS credentials found.")
        return None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "file_attachment_id" not in df.columns or "assign_class" not in df.columns:
            st.error("CSV must contain 'file_attachment_id' and 'assign_class' columns.")
            return None
        return df
    return None

def display_columns_selector(df):
    """Displays a multi-select for additional columns in the dataframe."""
    st.sidebar.header("Select Additional Columns to Display")
    additional_columns = st.sidebar.multiselect(
        "Select additional columns to display below",
        options=[col for col in df.columns if col not in ['file_attachment_id', 'assign_class', 'prediction', 'ddx', 'reasoning', 'morphology', 'user_message_input', 'simple_description']]
    )
    return additional_columns

def create_presigned_url(file_attachment_id, assign_class, expiration=3600):
    """Generates a presigned URL for accessing an image in S3."""
    bucket = "derm-data-general"
    key = f"DATASETS/TRAINING/cropped_images/v2_crop_256/{assign_class}/{file_attachment_id}_0.jpg"
    try:
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
        return response
    except NoCredentialsError:
        st.error("No AWS credentials found.")
        return None

def display_image(file_attachment_id, assign_class):
    """Displays the image with adjusted size and relevant metadata."""
    image_url = create_presigned_url(file_attachment_id, assign_class)
    if image_url:
        st.image(image_url, caption=f"{assign_class} - {file_attachment_id}", width=300)

def display_metadata(row, additional_columns):
    """Displays metadata below the image in two columns: GROUND TRUTH and MODEL PREDICTION, plus additional columns."""
    column_mapping = {
        'prediction': 'Model DDx',
        'reasoning': 'Summary',
        'user_message_input': 'Expert Term predictions',
        'ddx': 'Doctor DDx',
        'morphology': 'Doctor Description',
        'simple_description': 'Body Location'
    }

    # Split data into two sections
    ground_truth_columns = ['prediction', 'reasoning', 'user_message_input']
    model_prediction_columns = ['ddx', 'morphology', 'simple_description']

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MODEL PREDICTION")
        for col in ground_truth_columns:
            if col in row.index:
                st.write(f"**{column_mapping[col]}**")
                st.write(row[col])

    with col2:
        st.subheader("GROUND TRUTH")
        for col in model_prediction_columns:
            if col in row.index:
                st.write(f"**{column_mapping[col]}**")
                st.write(row[col])

    # Display additional columns below the two main sections
    if additional_columns:
        st.subheader("Additional Information")
        for col in additional_columns:
            if col in row.index:
                st.write(f"**{col}**")
                st.write(row[col])

def main():
    """Main function to run the application."""
    st.title("S3 Image Viewer with Metadata")

    df = upload_file()
    if df is not None:
        additional_columns = display_columns_selector(df)

        # Display the dataframe in a table format for easy row selection
        st.sidebar.header("Select Row")
        st.write("### Data Preview")
        st.dataframe(df)

        # Dropdown to select a row by file_attachment_id and assign_class
        row_selector = st.sidebar.selectbox(
            "Choose a row",
            options=df.apply(lambda row: f"{row['file_attachment_id']} - {row['assign_class']}", axis=1)
        )

        # Get the selected row based on the chosen dropdown option
        selected_index = df.apply(lambda row: f"{row['file_attachment_id']} - {row['assign_class']}", axis=1).tolist().index(row_selector)
        selected_row = df.iloc[selected_index]

        # Display the image
        display_image(selected_row["file_attachment_id"], selected_row["assign_class"])

        # Display the metadata below the image
        display_metadata(selected_row, additional_columns)

if __name__ == "__main__":
    main()

