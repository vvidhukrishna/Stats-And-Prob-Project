import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import *
file = '/Users/vvinodkrishnan/Desktop/code files/student_depression_dataset.csv'

"""
Questions:
Question 1:
"What is the probability that a student has depression given
they have had suicidal thoughts, and how does it compare to
the overall probability of depression? Also, apply Bayes’ 
theorem to estimate the probability that a student had 
suicidal thoughts given they are depressed."
Question 2:
"Which distribution best fits the CGPA of students? Estimate
its parameters and evaluate the goodness of fit. What 
proportion of students have CGPA above 8 based on this 
model?"
Question 3:
"Model the distribution of daily Work/Study Hours using a 
Poisson or other discrete distribution. Does higher daily 
workload correlate with higher depression probability?"
Question 4:
"Compare the distribution of Academic Pressure ratings 
between students with and without suicidal thoughts. Does 
the group with suicidal thoughts show a significantly 
different distribution?"
Question 5 (Module 3):
"Is there a significant difference in the average CGPA 
between students with and without depression?"
Question 6:
"What is the probability of a student having depression 
given they have a family history of mental illness?"
Question 7:
“Can we predict a student’s depression level based on 
multiple lifestyle and academic factors?”
Use multiple linear regression with predictors like:
Academic Pressure
CGPA
Sleep Duration
Work/Study Hours
Study Satisfaction
Financial Stress
Family History of Mental Illness

"""

'''
RangeIndex: 27901 entries, 0 to 27900
Data columns (total 18 columns):
 #   Column                                 Non-Null Count  Dtype  
---  ------                                 --------------  -----  
 0   id                                     27901 non-null  int64  
 1   Gender                                 27901 non-null  object 
 2   Age                                    27901 non-null  float64
 3   City                                   27901 non-null  object 
 4   Profession                             27901 non-null  object 
 5   Academic Pressure                      27901 non-null  float64
 6   Work Pressure                          27901 non-null  float64
 7   CGPA                                   27901 non-null  float64
 8   Study Satisfaction                     27901 non-null  float64
 9   Job Satisfaction                       27901 non-null  float64
 10  Sleep Duration                         27901 non-null  object 
 11  Dietary Habits                         27901 non-null  object 
 12  Degree                                 27901 non-null  object 
 13  Have you ever had suicidal thoughts ?  27901 non-null  object 
 14  Work/Study Hours                       27901 non-null  float64
 15  Financial Stress                       27901 non-null  object 
 16  Family History of Mental Illness       27901 non-null  object 
 17  Depression                             27901 non-null  int64  
dtypes: float64(7), int64(2), object(9)
memory usage: 3.8+ MB
None
'''

# question 3
df = pd.read_csv(file)
print(df["Work/Study Hours"].head(10))

hour_counts = df["Work/Study Hours"].value_counts().sort_index()
x = np.arange(0, 13)

mean = df["Work/Study Hours"].mean()
lambd = 1 / mean
print(lambd)
x_max = max(x)

exp = lambd * np.exp(-lambd * (x_max - x))
exp_scaled = exp * len(df)

plt.bar(hour_counts.index, hour_counts.values, alpha=0.6, label="Observed Frequency")
plt.plot(x, exp_scaled, color='purple', linewidth=2, label="Exponential (Visual Fit)")

plt.title("Work/Study Hours with Exponential Curve")

plt.xlabel("Work/Study Hours")
plt.ylabel("Frequency")

plt.xticks(range(0, 13))
plt.grid(axis='y')
plt.legend()
plt.show()

# question 4
df = pd.read_csv(file)

with_thoughts = df[df["Have you ever had suicidal thoughts ?"] == "Yes"]["Academic Pressure"]
without_thoughts = df[df["Have you ever had suicidal thoughts ?"] == "No"]["Academic Pressure"]

plt.figure(figsize=(10, 6))

sns.kdeplot(with_thoughts, label="With Suicidal Thoughts", shade=True)
sns.kdeplot(without_thoughts, label="Without Suicidal Thoughts", shade=True)

plt.title("Distribution of Academic Pressure Ratings")

plt.xlabel("Academic Pressure")
plt.ylabel("Density")

plt.legend()
plt.grid(True)
plt.show()


# question 5
stat, p = mannwhitneyu(with_thoughts, without_thoughts, alternative='two-sided')
print("Mann-Whitney U test p-value:", p)

cgpa_depressed = df[df['Depression'] == 1]['CGPA']
cgpa_not_depressed = df[df['Depression'] == 0]['CGPA']

mean_depressed = cgpa_depressed.mean()
std_depressed = cgpa_depressed.std()

mean_not_depressed = cgpa_not_depressed.mean()
std_not_depressed = cgpa_not_depressed.std()

print(mean_depressed, std_depressed, mean_not_depressed,
 std_not_depressed)

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Depression", y="CGPA")

plt.title("CGPA Distribution: With vs Without Depression")

plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("CGPA")

plt.grid(True)
plt.show()


# question 6
df = pd.read_csv(file)
family_history = df[df["Family History of Mental Illness"] == "Yes"]

p_depression_given_family_history = family_history["Depression"].mean()

print("P(Depression | Family History = Yes):", p_depression_given_family_history)

p_yes = df[df["Family History of Mental Illness"] == "Yes"]["Depression"].mean()
p_no = df[df["Family History of Mental Illness"] == "No"]["Depression"].mean()

plt.bar(["Family History = Yes", "Family History = No"], [p_yes, p_no], color=['orange', 'skyblue'])

plt.title("P(Depression | Family History of Mental Illness)")

plt.ylabel("Probability of Depression")
plt.ylim(0, 1)

plt.grid(axis='y')
plt.show()