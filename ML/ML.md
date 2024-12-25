# Machine Learning

Machine Learning (ML) is a branch of artificial intelligence (AI) that allows systems to learn from data and make predictions or decisions without being explicitly programmed. There are three main types of machine learning:

---

## 1. Reinforcement Learning
Reinforcement learning focuses on training an agent to make a sequence of decisions. The agent learns by interacting with an environment and receiving rewards or penalties based on its actions. 

**Examples**:
- A robot deciding its path to reach a destination.
- A chess engine determining the next optimal move.  

This approach is widely used in **game theory**, robotics, and autonomous systems.

---

## 2. Supervised Learning
In supervised learning, the model learns from a labeled dataset, where each input has a corresponding output (target variable). The goal is to predict the target variable for new, unseen data.

**Key Concepts**:
- **Target Variable (Labels)**: The output we want the model to predict.
- **Observations**: Rows or examples in the dataset, representing the instances the model learns from.
- **Features**: Columns or attributes that provide input information to help predict the target.

**Workflow**:
1. Provide a dataset with input features and corresponding target labels (training data).
2. Train the model to learn patterns from the data.
3. Use the trained model to make predictions on new, unseen inputs.

**Examples**:
- Predicting house prices based on features like size, location, and number of rooms.
- Classifying emails as spam or not spam.

---

## 3. Unsupervised Learning
In unsupervised learning, the model works with an unlabeled dataset. The goal is to identify patterns, structures, or groupings in the data without predefined labels.

**Key Concepts**:
- **Clustering**: Grouping data points into clusters based on similarity (e.g., customer segmentation).
- **Dimensionality Reduction**: Simplifying data while preserving important features (e.g., Principal Component Analysis).

**Examples**:
- Grouping customers with similar purchasing behavior.
- Detecting anomalies in network traffic for cybersecurity.

---

### Summary Table

| Type                  | Input Data        | Output/Goal                     | Examples                                |
|-----------------------|-------------------|----------------------------------|----------------------------------------|
| Reinforcement Learning | Interaction with an environment | Sequential decision-making        | Chess engines, robotics               |
| Supervised Learning    | Labeled data     | Predicting target variable       | Spam detection, price prediction       |
| Unsupervised Learning  | Unlabeled data   | Finding patterns and structures  | Customer segmentation, anomaly detection |

