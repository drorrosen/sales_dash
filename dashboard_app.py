import pandas as pd
import streamlit as st
from Functions import *  # Ensure this matches the name of the file and class where Preprocessing is defined
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import altair as alt
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO







#######################################
# PAGE SETUP
#######################################




# Set page config
st.set_page_config(
    page_title="Sales Dashboard",
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
        This app contains 1 dashboards - 
        1) General Dashboard - Please upload latest months sales data (according to the correct structure)
        """
    )


def dashboard_1():
    st.header("Sales Dashboard! 🚀")
    st.sidebar.success("Upload a file to get started.")

    df_uploaded = file_upload()
    if df_uploaded is not None:
        # Generate a unique session key for the uploaded file based on its name and upload time
        session_key = f"data_cleaned"

        # Check if the file or its processing is already in session state (to avoid reprocessing)
        if session_key not in st.session_state:
            # Process the uploaded data and store in session state
            processor = DataFrameProcessor(df_uploaded)  # Adjust if your class is designed differently
            processed_df = processor.process()

            content = download_file_from_dropbox('/all_months.xlsx')
            try:
                full_df = pd.read_excel(BytesIO(content))
            except:
                full_df = pd.read_csv(BytesIO(content))

            data_cleaned = pd.concat([full_df, processed_df], axis=0, ignore_index=True, sort=False)
            data_cleaned = data_cleaned.drop_duplicates()
            data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')

            # Store the cleaned data in session state using the unique session key
            st.session_state[session_key] = data_cleaned

        # Retrieve the processed and stored data from session state
        data_cleaned = st.session_state[session_key]




        #updating the data in dropbox
        upload_dataframe_to_dropbox(data_cleaned, '/all_months.xlsx')

        # Assuming data_cleaned is your DataFrame and it's already loaded
        filtered_data = data_cleaned.copy()

        # Filter the DataFrame to include only data from 2021 to 2023
        start_date = '2021-01-01'
        filtered_data = filtered_data[(filtered_data['Date'] >= start_date)]


        # Extract years and months
        filtered_data['Year'] = filtered_data['Date'].dt.year
        filtered_data['Month'] = filtered_data['Date'].dt.month

        # Get unique years and months for selection
        years = sorted(filtered_data['Year'].unique())
        months = sorted(filtered_data['Month'].unique())



        # Sidebar multiselect for years and months with the option to select none
        selected_years = st.sidebar.multiselect('Select Years', options=years, default=years)
        selected_months = st.sidebar.multiselect('Select Months', options=months, default=months)

        # Filter data based on the selected year and month
        # If no year or month is selected, it defaults to including all data
        if selected_years:
            filtered_data = filtered_data[filtered_data['Year'].isin(selected_years)]
        if selected_months:
            filtered_data = filtered_data[filtered_data['Month'].isin(selected_months)]
        # Display filtered data
        # Example: Display filtered data

        st.subheader("Filtered Data")
        st.write(filtered_data)









        # Assuming data_cleaned is your DataFrame and 'Gross Profit' is the column with potential text values
        filtered_data['Gross Profit'] = pd.to_numeric(filtered_data['Gross Profit'], errors='coerce')

        #calculate net profit
        filtered_data = calculate_net_profit(filtered_data)

        # add date lag values

        filtered_data = add_net_profit_lags(filtered_data, 7)


        # Apply the mapping function to the 'Category' column
        filtered_data['Received/Returned'] = filtered_data['Received/Returned'].str.lower().str.strip()
        filtered_data['Shipping Returns'] = filtered_data['Received/Returned'].apply(map_to_returned)

        # Now data_cleaned can be used for visualization and analysis
        #st.write("Cleaned Data", data_cleaned.head())


        st.markdown('## Records')
        # Display records held
        # These calculations assume your data has been processed to include all necessary fields
        col0_1, col_0_2 = st.columns(2)
        with col0_1:
            high_month_sale = filtered_data.groupby(['Year', 'Month'])['Total Price'].sum().idxmax()
            st.metric("Highest Single Sale Month", f"{high_month_sale[0]}-{high_month_sale[1]}")
            highest_gross_revenue_month = filtered_data.groupby(['Year', 'Month'])['Gross Profit'].sum().idxmax()
            st.metric("Highest Gross Profit Month", f"{highest_gross_revenue_month[0]}-{highest_gross_revenue_month[1]}")
            highest_net_revenue_month = filtered_data.groupby(['Year', 'Month'])['Net Profit'].sum().idxmax()
            st.metric("Highest Net Profit Month", f"{highest_net_revenue_month[0]}-{highest_net_revenue_month[1]}")

        with col_0_2:

            highest_single_sale = filtered_data['Total Price'].max()
            st.metric("Highest Single Sale", f"AED {highest_single_sale:,.2f}")
            highest_gross_profit = filtered_data['Gross Profit'].max()
            st.metric("Highest Gross Profit", f"AED {highest_gross_profit:,.2f}")
            highest_net_profit = filtered_data['Net Profit'].max()
            st.metric("Highest Net Profit", f"AED {highest_net_profit:,.2f}")








# Title for the statistics section
        st.markdown("## 📈 Descriptive Statistics")

        # Function for formatting numbers nicely
        def format_stat(number):
            return f"{number:,.2f}"

        # Unit Price Stats
        st.markdown("### Unit Price")
        col1, col1_1,col2, col2_1 = st.columns(4)
        with col1:
            st.metric("Average", f"AED {format_stat(filtered_data['Unit Price'].mean())}")
        with col1_1:
            st.metric("Median", f"AED {format_stat(filtered_data['Unit Price'].median())}")
        with col2:
            st.metric("Standard Deviation", f"AED {format_stat(filtered_data['Unit Price'].std())}")
        with col2_1:
            st.metric("Max", f"AED {format_stat(filtered_data['Unit Price'].max())}")

        # Total Price Stats
        st.markdown("### Total Price")
        col3, col3_1 ,col4, col4_1 = st.columns(4)
        with col3:
            st.metric("Average", f"AED {format_stat(filtered_data['Total Price'].mean())}")
        with col3_1:
            st.metric("Median", f"AED {format_stat(filtered_data['Total Price'].median())}")
        with col4:
            st.metric("Standard Deviation", f"AED {format_stat(filtered_data['Total Price'].std())}")
        with col4_1:
            st.metric("Max", f"AED {format_stat(filtered_data['Total Price'].max())}")

        # Delivery Fee Stats
        st.markdown("### Shipping Fee")
        col5, col5_1,col6, col6_1 = st.columns(4)
        with col5:
            st.metric("Average", f"AED {format_stat(filtered_data['Shipping Fee'].mean())}")
        with col5_1:
            st.metric("Median", f"AED {format_stat(filtered_data['Shipping Fee'].median())}")
        with col6:
            st.metric("Standard Deviation", f"AED{format_stat(filtered_data['Shipping Fee'].std())}")
        with col6_1:
            st.metric("Max", f"AED {format_stat(filtered_data['Shipping Fee'].max())}")

        # Gross Profit Stats
        st.markdown("### Gross Profit")
        col7, col7_1,col8, col8_1 = st.columns(4)
        with col7:
            st.metric("Average", f"AED {format_stat(filtered_data['Gross Profit'].mean())}")
        with col7_1:
            st.metric("Median", f"AED {format_stat(filtered_data['Gross Profit'].median())}")
        with col8:
            st.metric("Standard Deviation", f"AED {format_stat(filtered_data['Gross Profit'].std())}")
        with col8_1:
            st.metric("Max", f"AED {format_stat(filtered_data['Gross Profit'].max())}")

        # Commission Stats (Conditional)
        if 'Commission (AED )' in filtered_data.columns:
            st.markdown("### Commission (AED)")
            col9, col9_1,col10, col10_1 = st.columns(4)
            with col9:
                st.metric("Average", f"AED {format_stat(filtered_data['Commission (AED)'].mean())}")
            with col9_1:
                st.metric("Median", f"AED {format_stat(filtered_data['Commission (AED)'].median())}")
            with col10:
                st.metric("Standard Deviation", f"AED {format_stat(filtered_data['Commission (AED)'].std())}")
            with col10_1:
                st.metric("Max", f"AED {format_stat(filtered_data['Commission (AED)'].max())}")

        # Net Profit Stats
        st.markdown("### Net Profit")
        col11, col11_1,col12, col12_1 = st.columns(4)
        with col11:
            st.metric("Average", f"AED {format_stat(filtered_data['Net Profit'].mean())}")
        with col11_1:
            st.metric("Median", f"AED {format_stat(filtered_data['Net Profit'].median())}")
        with col12:
            st.metric("Standard Deviation", f"AED {format_stat(filtered_data['Net Profit'].std())}")
        with col12_1:
            st.metric("Max", f"AED {format_stat(filtered_data['Net Profit'].max())}")




        st.subheader("Data Visualization")

        # Create two columns for charts
        fig_col1, fig_col2, fig_col3 = st.columns(3)
        if 'Shipping Returns' in filtered_data.columns:
            with fig_col1:
                # Plot the pie chart for Delivery Returns
                delivery_returns_counts = filtered_data['Shipping Returns'].value_counts()
                fig_pie = px.pie(names=delivery_returns_counts.index, values=delivery_returns_counts.values,
                                 title='Distribution of Shipping Returns')
                st.write(fig_pie)

        with fig_col2:

            columns_to_aggregate = ['Net Profit', 'Gross Profit']


            # Aggregating data
            data_grouped = filtered_data.groupby('Date').agg({col: 'sum' for col in columns_to_aggregate}).reset_index()

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
            data_grouped = filtered_data.groupby('Date').agg({'Gross Profit':'sum', 'Net Profit':'sum'}).reset_index()

            # Calculate 'Profit Margin (%)', avoiding division by zero
            data_grouped['Profit Margin (%)'] = np.where(
             data_grouped['Gross Profit'] != 0,  # Condition
            (data_grouped['Net Profit'] / data_grouped['Gross Profit']) * 100,  # True: calculate profit margin
            np.nan)
            # Plot the profit margin over time
            fig = px.line(data_grouped, x='Date', y='Profit Margin (%)', title='Profit Margin Percentage Over Time (Net Profit/Gross Profit)')
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Profit Margin (%)')

            # Display the figure
            st.write(fig)


        fig_col1, fig_col2, fig_col3 = st.columns(3)


        # Convert 'Date' to ordinal to use as a feature for linear regression
        data_grouped = filtered_data.groupby('Date')[['Unit Price', 'Total Price', 'Shipping Fee', 'Gross Profit', 'Net Profit']].mean().reset_index()
        #data_grouped['DateOrdinal'] = data_grouped['Date'].apply(lambda x: x.toordinal())


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
            model = RandomForestRegressor()
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
            fig.add_trace(go.Scatter(x=data_grouped['Date'], y=data_grouped['Shipping Fee'], name='Shipping Fee',
                                     mode='lines+markers', yaxis='y2',
                                     marker=dict(color='FireBrick')))

            # Create axis objects
            fig.update_layout(
                xaxis=dict(title='Date'),
                yaxis=dict(title='Quantity', side='left', showgrid=False),
                yaxis2=dict(title='Shipping Fee', side='right', overlaying='y', showgrid=False),
                title='Trends in Quantity and Shipping Fee Over Time'
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
        excel_data = to_excel(filtered_data)

        # Sidebar download button
        st.sidebar.header("Download the cleaned data")
        st.sidebar.download_button(
            label="Download Excel with the cleaned data",
            data=excel_data,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.ms-excel"
        )
        #

# def dashboard_2():
#     st.header("Monthly Comparison Dashboard")
#     uploaded_files = st.sidebar.file_uploader("Choose Excel/CSV files", accept_multiple_files=True, type=['csv', 'xlsx'])
#
#     if uploaded_files:
#         data_frames = []
#         trend_data = {'Gross Profit': [], 'Net Profit': [], 'Shipping Fee': [], 'Unit Price': [], 'Total Price': []}
#
#         for uploaded_file in uploaded_files:
#             if uploaded_file.name.endswith('.xlsx'):
#                 df = pd.read_excel(uploaded_file)
#             else:
#                 df = pd.read_csv(uploaded_file)
#
#             preprocessing = Preprocessing(df)  # Your preprocessing steps
#             preprocessing.translate_column_names()
#             if 'Date' not in preprocessing.data.columns:
#                 continue
#             preprocessing.clean_data()
#             preprocessing.calculate_net_profit()
#             df = preprocessing.data_cleaned
#             df['Date'] = pd.to_datetime(df['Date'])
#             df['Month'] = df['Date'].dt.strftime('%Y-%m')  # For aggregation, if needed
#
#             data_frames.append(df)
#
#             # Prepare data for trend comparison
#             for metric in trend_data.keys():
#                 if metric in df.columns:
#                     trend_metric = df.groupby('Date')[metric].sum() if metric != 'Unit Price' else df.groupby('Date')[metric].mean()
#                     trend_data[metric].append({'name': uploaded_file.name, 'data': trend_metric})
#
#         # Now that we have a list of DataFrames, you can compare them
#         # For simplicity, let's assume we want to compare the sum of a 'Sales' column across all files
#
#         # Example of a simple comparison - adapt as needed
#         comparison_data = {
#             'Filename': [],
#             'Gross Profit': [],
#             'Net Profit': [],
#             'Shipping Fee': [],
#             'Unit Price': [],
#             'Total Price': [],
#             'Commission (AED)': []
#
#         }
#
#         for uploaded_file, df in zip(uploaded_files, data_frames):
#             comparison_data['Filename'].append(uploaded_file.name)
#             comparison_data['Gross Profit'].append(df['Gross Profit'].sum())
#             comparison_data['Net Profit'].append(df['Net Profit'].sum())
#             comparison_data['Shipping Fee'].append(df['Shipping Fee'].sum())
#             comparison_data['Unit Price'].append(df['Unit Price'].mean())
#             comparison_data['Total Price'].append(df['Total Price'].mean())
#             if 'Commission (AED)' in df.columns:
#                 comparison_data['Commission (AED)'].append(df['Commission (AED)'].sum())
#             else:
#                 df['Commission (AED)'] = 0
#                 comparison_data['Commission (AED)'].append(df['Commission (AED)'].sum())
#
#
#         comparison_df = pd.DataFrame(comparison_data)
#
#         cols = st.columns(1)
#
#         with cols[0]:
#             st.write("### Comparison Table - ")
#             st.dataframe(comparison_df)
#
#
#
#
#     # Display the comparison DataFrame
#
#         fig = go.Figure(data=[
#             go.Bar(name='Gross Profit', x=comparison_df['Filename'], y=comparison_df['Gross Profit'], marker_color='blue'),
#             go.Bar(name='Net Profit', x=comparison_df['Filename'], y=comparison_df['Net Profit'], marker_color='green'),
#             go.Bar(name='Shipping Fee', x=comparison_df['Filename'], y=comparison_df['Shipping Fee'], marker_color='red'),
#             go.Bar(name='Commission (AED)', x=comparison_df['Filename'], y=comparison_df['Commission (AED)'], marker_color='yellow')
#
#         ])
#
#
#
#         # Change the bar mode to group to display bars side by side
#         fig.update_layout(barmode='group', title='Gross and Net Profit Comparison')
#
#         cols = st.columns(1)
#         with cols[0]:
#             st.plotly_chart(fig)
#
#
#         # Adjusted Plotting Logic for Two Plots Per Column
#         num_plots = len(trend_data)
#         cols_per_row = 3  # Define how many plots per row you want
#         num_rows = (num_plots + cols_per_row - 1) // cols_per_row  # Calculate the number of rows needed
#
#         for row in range(num_rows):
#             cols = st.columns(cols_per_row)  # Create two columns for each row
#             for col_index in range(cols_per_row):
#                 plot_index = row * cols_per_row + col_index
#                 if plot_index < num_plots:  # Check if there's a metric to plot
#                     metric = list(trend_data.keys())[plot_index]
#                     data_list = trend_data[metric]
#                     fig = go.Figure()
#                     for data in data_list:
#                         fig.add_trace(go.Scatter(x=data['data'].index, y=data['data'], mode='lines+markers', name=data['name']))
#                     fig.update_layout(title=f'Trend for {metric}', xaxis_title='Normalized Index', yaxis_title=metric, legend_title='File Name')
#
#                     with cols[col_index]:  # Display the plot in the correct column
#                         st.plotly_chart(fig)
#
#
#




page_names_to_funcs = {
    "Introduction": intro,
    "General Dashboard": dashboard_1
    #"Monthly Comparison Dashboard": dashboard_2,

}

dashboard_name = st.sidebar.selectbox("Choose a dashboard", page_names_to_funcs.keys())
page_names_to_funcs[dashboard_name]()
#%%


