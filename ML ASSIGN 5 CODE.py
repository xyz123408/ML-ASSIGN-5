import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Function to take input from the user for the transactions
def get_transactions():
    transactions = []
    print("Enter transactions (type 'done' to finish):")
    while True:
        transaction = input("Enter a transaction (comma-separated items): ")
        if transaction.lower() == 'done':
            break
        items = transaction.strip().split(',')
        items = [item.strip() for item in items]  # Remove any extra spaces
        transactions.append(items)
    return transactions

# Function to take input thresholds from the user
def get_thresholds():
    min_support = float(input("Enter minimum support threshold (e.g., 0.4): "))
    min_confidence = float(input("Enter minimum confidence threshold (e.g., 0.6): "))
    min_lift = float(input("Enter minimum lift threshold (e.g., 1.2): "))
    return min_support, min_confidence, min_lift

# Step 1: Get transactions and thresholds from the user
transactions = get_transactions()
min_support, min_confidence, min_lift = get_thresholds()

# Step 2: Encode the dataset
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 3: Apply the Apriori algorithm with user-defined min_support
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Step 4: Filter frequent itemsets by minimum length (size of the itemset)
frequent_itemsets_filtered = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]

# Step 5: Generate association rules with user-defined min_confidence
rules = association_rules(frequent_itemsets_filtered, metric="confidence", min_threshold=min_confidence)

# Step 6: Filter rules by lift
filtered_rules = rules[rules['lift'] >= min_lift]

# Output the filtered frequent itemsets and association rules
print("\nFiltered Frequent Itemsets:")
print(frequent_itemsets_filtered)

print("\nFiltered Association Rules (Confidence >= {} and Lift >= {}):".format(min_confidence, min_lift))
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Step 7: Visualization of rules - Support vs. Confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_rules, x='support', y='confidence', hue='lift', palette='viridis', size='lift', sizes=(20, 200), legend=None)
plt.title('Support vs Confidence of Association Rules (Lift Colored)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()

# Step 8: Visualization of Lift vs. Confidence
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_rules, x='lift', y='confidence', hue='support', palette='coolwarm', size='support', sizes=(20, 200), legend=None)
plt.title('Lift vs Confidence of Association Rules (Support Colored)')
plt.xlabel('Lift')
plt.ylabel('Confidence')
plt.show()
