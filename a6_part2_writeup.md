# Assignment 6 Part 2 - Writeup

---

## Question 1: Feature Importance

Based on your house price model, rank the four features from most to least important. Explain how you determined this ranking.

**YOUR ANSWER:**

1. Most Important: Bedrooms: 6648.97
2. Bathrooms: 3858.90
3. Age: 950.35
4. Least Important: SquareFeet: 121.11

**Explanation:**
This ranking is determined using the absolute values of the coefficients from the model, which show the severity of each feature’s impact on price.

---

## Question 2: Interpreting Coefficients

Choose TWO features from your model and explain what their coefficients mean in plain English. For example: "Each additional bedroom increases the price by $___"

**Feature 1:**
Every extra bedroom adds approximately $6648.97 to the predicted house price.

**Feature 2:**
Each additional bathroom raises the price by about $3858.90.

---

## Question 3: Model Performance

What was your model's R² score? What does this tell you about how well your model predicts house prices? Is there room for improvement?

**YOUR ANSWER:**
0.9936. This shpows that the model accounts for 99.36% of the variation in house prices. The performance is great, with only little room for improvement.

---

## Question 4: Adding Features

If you could add TWO more features to improve your house price predictions, what would they be and why?

**Feature 1:**
Stories/Floors

**Why it would help:**
Homes with more stories typically have more useable space, which can massively raise their market value.

**Feature 2:**
Backyard (0 = No, 1 = Yes)

**Why it would help:**
Backyards are very desirable, especially for families, and can massively increase property value.

---

## Question 5: Model Trust

Would you trust this model to predict the price of a house with 6 bedrooms, 4 bathrooms, 3000 sq ft, and 5 years old? Why or why not? (Hint: Think about the range of your training data)

**YOUR ANSWER:**
No, because those values likely fall outside the data range used to train the model. Predicting beyond the original range, extrapolation, can be unreliable since other unseen factors may affect the results.
