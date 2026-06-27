## 🔢 Linear Regression from Scratch (Streamlit)

---

### Overview
This project is an interactive, educational web application built with Streamlit and Python. It demonstrates how Linear Regression and Gradient Descent work under the hood by implementing the algorithm entirely from scratch using NumPy—without relying on heavy machine learning libraries like `scikit-learn`. 

Users can tweak hyperparameters in real-time, generate synthetic data, or upload custom datasets to observe how the model learns and converges.

### Features
* **Built From Scratch:** The core `LinearRegression` class uses plain mathematics and NumPy to calculate gradients and update weights.
* **Interactive Tuning:** Use sidebar sliders to adjust the **Learning Rate** and the **Number of Iterations** to see how they impact model convergence.
* **Dynamic Data Generation:** Instantly generate synthetic scatter plot data with an adjustable number of data points.
* **Custom CSV Uploads:** Upload your own simple 2D datasets (X, y format) to test the algorithm on real data.
* **Real-time Visualizations:** Features a dynamic line chart tracking Mean Squared Error (Loss) over iterations, and a Matplotlib plot showing the line of best fit against the data points.
* **Live Metrics:** Instantly displays the calculated Slope (Weight), Intercept (Bias), and final MSE.

### Prerequisites
To run this application locally, you will need Python installed along with a few data science libraries. 

Install the required dependencies using pip:
`pip install streamlit numpy matplotlib`

### Usage
1. Clone this repository to your local machine.
2. Navigate to the project folder in your terminal.
3. Run the Streamlit application using the following command:
   `streamlit run app.py`
4. A new tab will automatically open in your default web browser hosting the interactive UI.
5. Use the sidebar to switch between Synthetic Data and CSV Uploads, and adjust the sliders to watch the gradient descent algorithm adapt in real-time.

### CSV Upload Format
If you choose to upload your own data, ensure your `.csv` file contains exactly two columns with no headers:
* **Column 1:** Independent variable (X)
* **Column 2:** Dependent variable (y)
