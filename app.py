import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Linear Regression Class
# -------------------------------
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.weights = np.random.randn(n_features)
        self.bias = 0

        y = y.ravel()
        cost_history = []

        for _ in range(self.n_iters):
            # predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # cost
            cost = np.mean((y_pred - y) ** 2)
            cost_history.append(cost)

        return cost_history

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔢 Linear Regression from Scratch")
st.write("Upload data or generate synthetic data and observe gradient descent!")

# Sidebar
st.sidebar.header("⚙️ Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
iterations = st.sidebar.slider("Iterations", 100, 2000, 1000)

data_choice = st.sidebar.radio("Select Data Source", ["Synthetic Data", "Upload CSV"])

# -------------------------------
# Data Handling
# -------------------------------
if data_choice == "Synthetic Data":
    np.random.seed(42)
    n_points = st.sidebar.slider("Number of Points", 50, 500, 100)

    X_raw = 10 * np.random.rand(n_points, 1)
    y = 3 + 2 * X_raw + np.random.randn(n_points, 1) * 2

else:
    file = st.sidebar.file_uploader("Upload CSV (format: X,y)", type="csv")

    if file is not None:
        data = np.loadtxt(file, delimiter=',')
        X_raw = data[:, 0:1]
        y = data[:, 1:2]
    else:
        st.info("Please upload a CSV file or switch to synthetic data.")
        st.stop()

# -------------------------------
# Model Training
# -------------------------------
X_bias = np.c_[np.ones((len(X_raw), 1)), X_raw]

model = LinearRegression(learning_rate, iterations)
loss_values = model.fit(X_bias, y)

# -------------------------------
# Metrics Display
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric("Slope (w)", f"{model.weights[1]:.2f}")
    st.metric("Intercept (b)", f"{model.bias:.2f}")
    st.metric("Final MSE", f"{loss_values[-1]:.2f}")

with col2:
    st.line_chart(loss_values, use_container_width=True)
    st.caption("Loss vs Iterations")

# -------------------------------
# Visualization
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X_raw, y, color='blue', label='Actual Data')

y_predicted = model.predict(X_bias)
ax.plot(X_raw, y_predicted, color='red', label='Regression Line')

ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()

st.pyplot(fig)

# -------------------------------
# Final Message
# -------------------------------
st.success("✅ Gradient Descent Converged! Try changing parameters.")