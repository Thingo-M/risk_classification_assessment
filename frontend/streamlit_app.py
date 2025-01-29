import streamlit as st
import pandas as pd
import requests

# Add some documentation
st.write("---")
st.write("### Data Explorer")

# Load and cache the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r'c:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\accepted_2007_to_2018q4\accepted_2007_to_2018Q4.csv')
        #df = pd.read_csv(r'c:\Users\mafaesami\.cache\kagglehub\datasets\wordsforthewise\lending-club\versions\3\accepted_2007_to_2018q4\accepted_2007_to_2018Q4_reduced.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Show first 4 rows
    st.write("#### Sample Data (First 4 rows)")
    st.dataframe(df.head(4))

    # Add ID search functionality
    st.write("#### Search by ID")
    search_id = st.text_input("Enter ID")
    if search_id:
        try:
            search_id = int(search_id)
            # Make API call to backend
            response = requests.post(
                'http://127.0.0.1:8000/predict_by_id',
                json={'id': search_id}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.write("#### Risk Assessment for ID:", search_id)
                st.write(f"Probability of Default: {result['default_probability']:.2%}")
                if result['prediction'] == 1:
                    st.error("High Risk: Likely to Default")
                else:
                    st.success("Low Risk: Unlikely to Default")
            elif response.status_code == 404:
                st.warning("No entry found with this ID")
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except ValueError:
            st.error("Please enter a valid numeric ID")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the server: {str(e)}")

# Add some documentation
with st.expander("About this app"):
    st.write("This app predicts loan approval based on various factors.")