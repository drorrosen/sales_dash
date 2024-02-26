import streamlit as st
from Functions import *  # Ensure this matches the name of the file and class where Preprocessing is defined
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import altair as alt
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.linear_model import Ridge
from io import BytesIO


#######################################
# PAGE SETUP
#######################################




# Set page config
st.set_page_config(
    page_title="US Population Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

#alt.themes.enable("dark")


def intro():
    import streamlit as st

    st.write("# Welcome to your Sales Dashboard App! 👋")
    st.sidebar.success("Select a Dashboard above.")

    st.markdown(
        """
        This app contains 2 dashboards - 
        1) individual  monthly dashboard where you can check the statistics of 1 file
        2) combined dashboard where you can compare the statistics of multiple files 
        """
    )


def dashboard_1():
    st.header("Sales Dashboard! 🚀")
    st.sidebar.success("Upload a file to get started.")

    df_uploaded = file_upload()
    if df_uploaded is not None:
        # Use the Preprocessing class to clean the uploaded data
        preprocessing = Preprocessing(df_uploaded)  # Adjust this if your Preprocessing class is designed differently
        preprocessing.translate_column_names()
        preprocessing.clean_data()
        preprocessing.calculate_net_profit()

        data_cleaned = preprocessing.data_cleaned
        # Now data_cleaned can be used for visualization and analysis
        #st.write("Cleaned Data", data_cleaned.head())

        # Title for the statistics section
        st.markdown("## 📈 Descriptive Statistics")

        # Function for formatting numbers nicely
        def format_stat(number):
            return f"{number:,.2f}"

        # Unit Price Stats
        st.markdown("### Unit Price")
        col1, col1_1,col2, col2_1 = st.columns(4)
        with col1:
            st.metric("Average", f"AED {format_stat(data_cleaned['Unit Price'].mean())}")
        with col1_1:
            st.metric("Median", f"AED {format_stat(data_cleaned['Unit Price'].median())}")
        with col2:
            st.metric("Standard Deviation", f"AED {format_stat(data_cleaned['Unit Price'].std())}")
        with col2_1:
            st.metric("Max", f"AED {format_stat(data_cleaned['Unit Price'].max())}")

        # Total Price Stats
        st.markdown("### Total Price")
        col3, col3_1 ,col4, col4_1 = st.columns(4)
        with col3:
            st.metric("Average", f"AED {format_stat(data_cleaned['Total Price'].mean())}")
        with col3_1:
            st.metric("Median", f"AED {format_stat(data_cleaned['Total Price'].median())}")
        with col4:
            st.metric("Standard Deviation", f"AED {format_stat(data_cleaned['Total Price'].std())}")
        with col4_1:
            st.metric("Max", f"AED {format_stat(data_cleaned['Total Price'].max())}")

        # Delivery Fee Stats
        st.markdown("### Delivery Fee")
        col5, col5_1,col6, col6_1 = st.columns(4)
        with col5:
            st.metric("Average", f"AED {format_stat(data_cleaned['Delivery Fee'].mean())}")
        with col5_1:
            st.metric("Median", f"AED {format_stat(data_cleaned['Delivery Fee'].median())}")
        with col6:
            st.metric("Standard Deviation", f"AED{format_stat(data_cleaned['Delivery Fee'].std())}")
        with col6_1:
            st.metric("Max", f"AED {format_stat(data_cleaned['Delivery Fee'].max())}")

        # Gross Profit Stats
        st.markdown("### Gross Profit")
        col7, col7_1,col8, col8_1 = st.columns(4)
        with col7:
            st.metric("Average", f"AED {format_stat(data_cleaned['Gross Profit'].mean())}")
        with col7_1:
            st.metric("Median", f"AED {format_stat(data_cleaned['Gross Profit'].median())}")
        with col8:
            st.metric("Standard Deviation", f"AED {format_stat(data_cleaned['Gross Profit'].std())}")
        with col8_1:
            st.metric("Max", f"AED {format_stat(data_cleaned['Gross Profit'].max())}")

        # Commission Stats (Conditional)
        if 'Commission (AED )' in data_cleaned.columns:
            st.markdown("### Commission (AED)")
            col9, col9_1,col10, col10_1 = st.columns(4)
            with col9:
                st.metric("Average", f"AED {format_stat(data_cleaned['Commission (AED)'].mean())}")
            with col9_1:
                st.metric("Median", f"AED {format_stat(data_cleaned['Commission (AED)'].median())}")
            with col10:
                st.metric("Standard Deviation", f"AED {format_stat(data_cleaned['Commission (AED)'].std())}")
            with col10_1:
                st.metric("Max", f"AED {format_stat(data_cleaned['Commission (AED)'].max())}")

        # Net Profit Stats
        st.markdown("### Net Profit")
        col11, col11_1,col12, col12_1 = st.columns(4)
        with col11:
            st.metric("Average", f"AED {format_stat(data_cleaned['Net Profit'].mean())}")
        with col11_1:
            st.metric("Median", f"AED {format_stat(data_cleaned['Net Profit'].median())}")
        with col12:
            st.metric("Standard Deviation", f"AED {format_stat(data_cleaned['Net Profit'].std())}")
        with col12_1:
            st.metric("Max", f"AED {format_stat(data_cleaned['Net Profit'].max())}")




        st.subheader("Data Visualization")

        # Create two columns for charts
        fig_col1, fig_col2, fig_col3 = st.columns(3)
        if 'Delivery Returns' in data_cleaned.columns:
            with fig_col1:
                # Plot the pie chart for Delivery Returns
                delivery_returns_counts = data_cleaned['Delivery Returns'].value_counts()
                fig_pie = px.pie(names=delivery_returns_counts.index, values=delivery_returns_counts.values,
                                 title='Distribution of Delivery Returns')
                st.write(fig_pie)

        with fig_col2:

            columns_to_aggregate = ['Net Profit', 'Gross Profit']


            # Aggregating data
            data_grouped = data_cleaned.groupby('Date').agg({col: 'sum' for col in columns_to_aggregate}).reset_index()

            # Assuming 'Total Profit' is already calculated and included in `data_grouped`
            # If not, adjust the aggregation function accordingly

            # Melting the DataFrame for Plotly
            data_melted = data_grouped.melt(id_vars=['Date'], value_vars=columns_to_aggregate, var_name='Metric', value_name='Value')

            # Creating the area chart
            fig_area = px.line(data_melted, x='Date', y='Value', color='Metric', line_group='Metric',
                               title='Trends Over Time for Net Profit and Gross Profit')
            fig_area.update_xaxes(title_text='Date')
            fig_area.update_yaxes(title_text='Value in USD')

            # Displaying the figure in the Streamlit app
            st.plotly_chart(fig_area)






        # Assuming 'Date' in your DataFrame is already in datetime format.
        # If your data is not aggregated, aggregate 'Net Profit' and 'Total Price' by 'Date' (daily aggregation shown here).
        with fig_col3:
            data_grouped = data_cleaned.groupby('Date').agg({'Gross Profit':'sum', 'Net Profit':'sum'}).reset_index()

            # Calculate profit margin percentage for the aggregated data
            data_grouped['Profit Margin (%)'] = (data_grouped['Net Profit'] / data_grouped['Gross Profit'])

            # Plot the profit margin over time
            fig = px.line(data_grouped, x='Date', y='Profit Margin (%)', title='Profit Margin Percentage Over Time (Net Profit/Gross Profit)')
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Profit Margin (%)')

            # Display the figure
            st.write(fig)


        fig_col1, fig_col2, fig_col3 = st.columns(3)


        # Convert 'Date' to ordinal to use as a feature for linear regression
        data_grouped = data_cleaned.groupby('Date')[['Unit Price', 'Total Price', 'Delivery Fee', 'Gross Profit', 'Net Profit']].mean().reset_index()
        data_grouped['DateOrdinal'] = data_grouped['Date'].apply(lambda x: x.toordinal())


        with fig_col3:

            # Add 14 lags for 'Gross Profit'
            data_with_lags = add_net_profit_lags(data_grouped.copy(), 7)

            # Drop rows with NaN values created by shifting for lags
            data_with_lags.dropna(inplace=True)

            # Extract features and labels
            features = [f'Net_Profit_lag_{i}' for i in range(1, 8)]
            X = data_with_lags[features].values
            y = data_with_lags['Net Profit'].values

            # Initialize and train the model
            model = Ridge()
            model.fit(X, y)

            # Prepare to forecast the next 30 days, using a rolling approach
            predictions = []
            for _ in range(30):
                # Use the last 7 gross profit values to predict the next one
                X_next = np.array([y[-7:]]).reshape(1, -1)
                next_gross_profit = model.predict(X_next)[0]

                # Append the prediction
                predictions.append(next_gross_profit)

                # Update the y array to include this new prediction
                y = np.append(y, next_gross_profit)

            # Generating future dates
            last_date = data_grouped['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]

            # Visualizing the forecast
            fig = go.Figure()

            # Actual Gross Profit
            fig.add_trace(go.Scatter(x=data_grouped['Date'], y=data_grouped['Net Profit'], mode='lines+markers', name='Actual Net Profit'))

            # Forecasted Gross Profit
            fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines+markers', name='Forecasted Net Profit', line=dict(dash='dash')))

            fig.update_layout(title='Rolling Forecast for Net Profit',
                              xaxis_title='Date',
                              yaxis_title='Net Profit',
                              legend_title='Metric')

            st.plotly_chart(fig)



        with fig_col1:
            # Create a figure with secondary y-axis
            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(x=data_grouped['Date'], y=data_grouped ['Unit Price'], name='Unit Price', mode='lines+markers', yaxis='y1'))
            fig.add_trace(go.Scatter(x=data_grouped['Date'], y=data_grouped['Total Price'], name='Total Price', mode='lines+markers', yaxis='y2'))

            # Create axis objects
            fig.update_layout(
                xaxis=dict(title='Date'),
                yaxis=dict(title='Unit Price', side='left', showgrid=False),
                yaxis2=dict(title='Total Price', side='right', overlaying='y', showgrid=False),
                title='Trends in Unit Price and Total Price Over Time'
            )

            # Add legend and layout options as needed
            fig.update_layout(legend=dict(x=0.01, y=0.99, traceorder='normal', font=dict(size=12)))

            # Display the figure in Streamlit
            st.plotly_chart(fig)



        # Ensure 'Total Price' and 'Unit Price' are not zero or null to avoid division by zero errors
        data_grouped = data_grouped[(data_grouped['Unit Price'] > 0) & (data_grouped['Total Price'] > 0)]

        # Calculate Quantity
        data_grouped['Quantity'] = data_grouped['Total Price'] / data_grouped['Unit Price']

        # Since Quantity is typically an integer, you might want to round or floor the values
        data_grouped['Quantity'] = data_grouped['Quantity'].apply(lambda x: round(x))

        with fig_col2:

            # Create a figure with secondary y-axis for Delivery Fee
            fig = go.Figure()

            # Add Delivery Fee trace on secondary y-axis
            fig.add_trace(go.Scatter(x=data_grouped['Date'], y=data_grouped['Delivery Fee'], name='Delivery Fee',
                                     mode='lines+markers', yaxis='y2',
                                     marker=dict(color='FireBrick')))

            # Create axis objects
            fig.update_layout(
                xaxis=dict(title='Date'),
                yaxis=dict(title='Quantity', side='left', showgrid=False),
                yaxis2=dict(title='Delivery Fee', side='right', overlaying='y', showgrid=False),
                title='Trends in Quantity and Delivery Fee Over Time'
            )

            # Add legend and layout options as needed
            fig.update_layout(legend=dict(x=0.01, y=0.99, traceorder='normal', font=dict(size=12)))

            # Display the figure
            st.plotly_chart(fig)












### Download the results

        # Assuming 'data' is the DataFrame you want users to be able to download

        def to_excel(df):
            """
            Convert the DataFrame into an Excel file stored in a BytesIO object,
            allowing for download in Streamlit.
            """
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
                #writer.save()
            processed_data = output.getvalue()
            return processed_data

        # Convert DataFrame to Excel in-memory
        excel_data = to_excel(data_grouped)

        # Sidebar download button
        st.sidebar.header("Download Excel file with results grouped by date")
        st.sidebar.download_button(
            label="Download Excel file with results",
            data=excel_data,
            file_name="processed_data.xlsx",
            mime="application/vnd.ms-excel"
        )


        # Convert DataFrame to Excel in-memory
        excel_data = to_excel(data_cleaned)

        # Sidebar download button
        st.sidebar.header("Download the cleaned data")
        st.sidebar.download_button(
            label="Download Excel with the cleaned data",
            data=excel_data,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.ms-excel"
        )
        #

def dashboard_2():
    st.header("Monthly Comparison Dashboard")
    uploaded_files = st.sidebar.file_uploader("Choose Excel/CSV files", accept_multiple_files=True, type=['csv', 'xlsx'])

    if uploaded_files:
        data_frames = []
        trend_data = {'Gross Profit': [], 'Net Profit': [], 'Delivery Fee': [], 'Unit Price': [], 'Total Price': []}

        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            preprocessing = Preprocessing(df)  # Your preprocessing steps
            preprocessing.translate_column_names()
            if 'Date' not in preprocessing.data.columns:
                continue
            preprocessing.clean_data()
            preprocessing.calculate_net_profit()
            df = preprocessing.data_cleaned
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.strftime('%Y-%m')  # For aggregation, if needed

            data_frames.append(df)

            # Prepare data for trend comparison
            for metric in trend_data.keys():
                if metric in df.columns:
                    trend_metric = df.groupby('Date')[metric].sum() if metric != 'Unit Price' else df.groupby('Date')[metric].mean()
                    trend_data[metric].append({'name': uploaded_file.name, 'data': trend_metric})

        # Now that we have a list of DataFrames, you can compare them
        # For simplicity, let's assume we want to compare the sum of a 'Sales' column across all files

        # Example of a simple comparison - adapt as needed
        comparison_data = {
            'Filename': [],
            'Gross Profit': [],
            'Net Profit': [],
            'Delivery Fee': [],
            'Unit Price': [],
            'Total Price': [],
            'Commission (AED)': []

        }

        for uploaded_file, df in zip(uploaded_files, data_frames):
            comparison_data['Filename'].append(uploaded_file.name)
            comparison_data['Gross Profit'].append(df['Gross Profit'].sum())
            comparison_data['Net Profit'].append(df['Net Profit'].sum())
            comparison_data['Delivery Fee'].append(df['Delivery Fee'].sum())
            comparison_data['Unit Price'].append(df['Unit Price'].mean())
            comparison_data['Total Price'].append(df['Total Price'].mean())
            if 'Commission (AED)' in df.columns:
                comparison_data['Commission (AED)'].append(df['Commission (AED)'].sum())
            else:
                df['Commission (AED)'] = 0
                comparison_data['Commission (AED)'].append(df['Commission (AED)'].sum())


        comparison_df = pd.DataFrame(comparison_data)



        # Display the comparison DataFrame
        st.write(comparison_df)

        fig = go.Figure(data=[
            go.Bar(name='Gross Profit', x=comparison_df['Filename'], y=comparison_df['Gross Profit'], marker_color='blue'),
            go.Bar(name='Net Profit', x=comparison_df['Filename'], y=comparison_df['Net Profit'], marker_color='green'),
            go.Bar(name='Delivery Fee', x=comparison_df['Filename'], y=comparison_df['Delivery Fee'], marker_color='red'),
            go.Bar(name='Commission (AED)', x=comparison_df['Filename'], y=comparison_df['Commission (AED)'], marker_color='yellow')

        ])



        # Change the bar mode to group to display bars side by side
        fig.update_layout(barmode='group', title='Gross and Net Profit Comparison')

        st.plotly_chart(fig)


        # Plotting trend comparisons for each metric
        for metric, data_list in trend_data.items():
            fig = go.Figure()
            for data in data_list:
                fig.add_trace(go.Scatter(x=data['data'].index, y=data['data'], mode='lines+markers', name=data['name']))
            fig.update_layout(title=f'Trend for {metric}', xaxis_title='Date', yaxis_title=metric, legend_title='File Name')
            st.plotly_chart(fig)







page_names_to_funcs = {
    "Introduction": intro,
    "Individual Monthly Analysis Dashboard": dashboard_1,
    "Monthly Comparison Dashboard": dashboard_2,

}

dashboard_name = st.sidebar.selectbox("Choose a dashboard", page_names_to_funcs.keys())
page_names_to_funcs[dashboard_name]()
#%%


