# CodeAlpha_Task

## 🔹 Step 1: Import Libraries

The required Python libraries are imported:

* **pandas, numpy** → Data manipulation
* **matplotlib, seaborn** → Data visualization
* **requests, StringIO** → Load dataset from a URL
* **scipy.stats** → Statistical testing
* **warnings** → Ignore unnecessary warnings

## 🔹 Step 2: Load Dataset with Retry

```python
def load_data(url, max_retries=3):
```

* Tries to fetch the dataset from GitHub.
* If it fails, retries 3 times.
* If still fails, loads from a **fallback URL** (Stanford server).
* This makes the code **robust to internet issues**.

## 🔹 Step 3: Perform EDA Function

```python
def perform_eda(data):
```

This function does the complete Exploratory Data Analysis.

### (a) **Basic Data Inspection**

* Shows first 5 rows (`head`)
* Prints dataset info (datatypes, null values)
* Descriptive statistics (`describe`)
### (b) **Missing Values Analysis**

* Detects missing values.
* Handles them:

  * Fill `Age` with **median**
  * Fill `Embarked` with **mode**
  * Drop `Cabin` column (too many missing values)

### (c) **Visualizations**

Creates **6 plots** using Matplotlib + Seaborn:

1. **Age Distribution** → Histogram with KDE
2. **Survival by Gender** → Barplot
3. **Survival by Class** → Barplot
4. **Family Size vs Survival** → Barplot (`FamilySize = SibSp + Parch + 1`)
5. **Fare vs Survival** → Boxplot
6. **Feature Correlation Heatmap** → Correlation between numeric variables

(d) **Statistical Testing**

Two tests are performed:

1. **T-test** → Compares average Age of survivors vs non-survivors
2. **Chi-square test** → Tests relationship between Passenger Class and Survival

 (e) **Key Insights**

Prints summary points such as:

* Overall survival rate
* Female survival rate
* 1st class survival rate
* Average passenger age
* Correlation of Fare with survival

 Step 4: Main Execution

```python
if __name__ == "__main__":
```

* Loads Titanic dataset using `load_data()`
* Runs `perform_eda()`
* Prints results + visualizations
* <img width="1798" height="729" alt="Screenshot 2025-08-19 195653" src="https://github.com/user-attachments/assets/32fcf876-4daf-4c01-9fa3-b1200e3415ef" />

<img width="1290" height="542" alt="Screenshot 2025-08-19 195712" src="https://github.com/user-attachments/assets/b2c9e7d6-3950-4bb8-9bfc-e65f6a2029c0" />

<img width="1615" height="638" alt="Screenshot 2025-08-19 195750" src="https://github.com/user-attachments/assets/2f7de6d9-c0e0-40eb-9831-a24f06fd2b74" />

<img width="1755" height="471" alt="Screenshot 2025-08-19 195814" src="https://github.com/user-attachments/assets/3e3021f8-322a-4e67-8007-9118aac0b718" />




