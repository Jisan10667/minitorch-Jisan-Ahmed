"""
Streamlit app for visualizing MiniTorch operations.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("MiniTorch Module 0 - Mathematical Operators")

st.sidebar.header("Function Visualizer")

# Import your operators (once implemented)
try:
    from minitorch.operators import sigmoid, relu, add, mul
    operators_available = True
except (ImportError, NotImplementedError):
    st.error("Operators not yet implemented. Complete Task 0.1 first!")
    operators_available = False

if operators_available:
    func = st.sidebar.selectbox("Select function to visualize:", 
                               ["sigmoid", "relu"])
    
    x = np.linspace(-5, 5, 100)
    
    try:
        if func == "sigmoid":
            y = [sigmoid(xi) for xi in x]
            st.write("## Sigmoid Function")
            st.write("Formula: σ(x) = 1 / (1 + e^(-x))")
            
        elif func == "relu":
            y = [relu(xi) for xi in x] 
            st.write("## ReLU Function")
            st.write("Formula: ReLU(x) = max(0, x)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel(f'{func}(x)', fontsize=12)
        ax.set_title(f'{func.title()} Function', fontsize=14)
        st.pyplot(fig)
        
    except NotImplementedError:
        st.error(f"{func} function not yet implemented!")

st.write("## Test Your Operators")

if operators_available:
    col1, col2 = st.columns(2)
    
    with col1:
        a = st.number_input("Enter first number:", value=2.0)
        b = st.number_input("Enter second number:", value=3.0)
        
    with col2:
        try:
            if st.button("Test Add"):
                result = add(a, b)
                st.success(f"add({a}, {b}) = {result}")
        except NotImplementedError:
            st.error("Add function not implemented!")
            
        try:
            if st.button("Test Multiply"):  
                result = mul(a, b)
                st.success(f"mul({a}, {b}) = {result}")
        except NotImplementedError:
            st.error("Multiply function not implemented!")
            
        try:
            if st.button("Test Sigmoid"):
                result = sigmoid(a)
                st.success(f"sigmoid({a}) = {result:.4f}")
        except NotImplementedError:
            st.error("Sigmoid function not implemented!")
            
        try:
            if st.button("Test ReLU"):
                result = relu(a)
                st.success(f"relu({a}) = {result}")
        except NotImplementedError:
            st.error("ReLU function not implemented!")

st.write("---")
st.write("### Testing Progress")

# Show testing progress
try:
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', '--tb=no', '-q'], 
                          capture_output=True, text=True, cwd='.')
    if result.returncode == 0:
        st.success("All tests passing!")
    else:
        st.warning("Some tests still failing. Keep implementing!")
except:
    st.info("Run `pytest` in terminal to check your progress")

st.write("### Tasks Completed")
st.write("- [ ] Task 0.1: Mathematical Operators")  
st.write("- [ ] Task 0.2: Property Testing")
st.write("- [ ] Task 0.3: Functional Programming")
st.write("- [ ] Task 0.4: Module System")
