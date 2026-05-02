# MiniTorch Module 0: ML Programming Foundations

Welcome to MiniTorch! You'll build a deep learning library from scratch to understand how frameworks like PyTorch work internally.

## Project Structure

```
minitorch/
├── minitorch/           # Main package
│   ├── operators.py     # Mathematical operations (Task 0.1)
│   ├── module.py        # Parameter management (Task 0.4)  
│   └── testing.py       # Utility functions
├── tests/               # Test suite
├── project/             # Demo applications
└── requirements.txt     # Dependencies
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Run tests (will fail initially):**
   ```bash
   pytest -m task0_1 -v
   ```

3. **Launch visualization:**
   ```bash
   streamlit run project/app.py
   ```

4. **Implement functions in:**
   - `minitorch/operators.py` (Tasks 0.1 & 0.3)
   - `minitorch/module.py` (Task 0.4)

## Tasks

- **Task 0.1**: Mathematical operators (add, mul, sigmoid, relu, etc.)
- **Task 0.2**: Property-based testing (already implemented)
- **Task 0.3**: Functional programming (map, reduce, zipWith)
- **Task 0.4**: Module system for parameters

## Testing

```bash
# Test specific task
pytest -m task0_1 -xvs

# Test all tasks
pytest -m "task0_1 or task0_2 or task0_3 or task0_4" -v

# Check code style
black minitorch/ tests/
flake8 minitorch/ tests/
```

## Success Criteria

- All tests passing
- Streamlit app works
- Ready for Module 1!

---

*Good luck building your deep learning library!*
