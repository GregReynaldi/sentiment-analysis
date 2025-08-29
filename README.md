# Sentiment Analysis with Ensemble Learning

A Python-based sentiment analysis system that uses multiple machine learning models to classify text reviews as positive or negative. The system employs ensemble learning with voting to achieve higher accuracy and robustness.

## 🚀 Features

- **Multi-Model Ensemble**: Uses 6 different machine learning algorithms for robust predictions
- **Advanced Text Processing**: Implements N-grams, lemmatization, and stopword removal
- **Web Scraping Integration**: Automatically fetches training data from online sources
- **Voting System**: Combines predictions from multiple models for improved accuracy
- **Easy-to-Use Interface**: Simple command-line interface for quick predictions

## 🧠 Machine Learning Models

The system uses the following models for ensemble learning:

1. **Naive Bayes** (Original NLTK implementation)
2. **Support Vector Machine** (Linear SVC)
3. **Multinomial Naive Bayes**
4. **Bernoulli Naive Bayes**
5. **Logistic Regression**
6. **Stochastic Gradient Descent**

## 📁 Project Structure

```
sentiment-analysis/
├── main.py              # Main prediction script
├── train.py             # Model training script
├── pre_process.py       # Text preprocessing utilities
├── scrape.py           # Web scraping functionality
├── store_params.py     # Model serialization utilities
├── requirements.txt    # Python dependencies
├── data.txt           # Vocabulary features
├── train.ipynb        # Jupyter notebook for training [same like train.py]
└── models/            # Serialized trained models
    ├── original.pickle
    ├── lin_svc.pickle
    ├── mnb.pickle
    ├── bnb.pickle
    ├── log_r.pickle
    └── sgd.pickle
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GregReynaldi/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 📖 Usage

### Making Predictions

Once the models are trained and have pickle file in models folder, you can make predictions:

```bash
python main.py
```

The system will prompt you to enter a review text, and it will:
- Preprocess your input text
- Run predictions through all 6 models
- Use majority voting to determine the final prediction
- Display the result with confidence percentage

### Training the Models

If u want to test and train by your own :

```bash
python train.py
```

This will:
- Scrape positive and negative review data from online sources
- Preprocess the text data
- Train 6 different machine learning models
- Save the trained models to the `models/` directory

### Example

```
What is the reviews that gonna be predict: The cinematography was stunning, and every scene felt alive with detail — it’s definitely a film worth watching again.

This Review is POSITIVE review with 100% Sure
```

```
What is the reviews that gonna be predict: The pacing was painfully slow, and the movie dragged on far longer than it needed to

This Review is NEGATIVE review with 100% Sure
```

## 🔧 Technical Details

### Text Preprocessing Pipeline

1. **Lowercasing**: Convert all text to lowercase
2. **N-gram Processing**: Handle negations (e.g., "not good" → "not_good")
3. **Tokenization**: Split text into individual words
4. **Stopword Removal**: Remove common English stopwords
5. **Lemmatization**: Reduce words to their root forms
6. **Feature Extraction**: Convert to bag-of-words representation

### Model Training Process

1. **Data Collection**: Scrape positive and negative reviews
2. **Preprocessing**: Apply the text preprocessing pipeline
3. **Feature Selection**: Use the 3000 most common words as features
4. **Model Training**: Train 6 different classifiers
5. **Model Serialization**: Save trained models using pickle

### Prediction Process

1. **Input Preprocessing**: Apply same preprocessing as training data
2. **Feature Mapping**: Convert input to feature vector
3. **Multi-Model Prediction**: Get predictions from all 6 models
4. **Ensemble Voting**: Use majority vote for final decision
5. **Confidence Calculation**: Calculate percentage based on model agreement

## 🎯 Performance

The system uses ensemble learning to achieve robust performance:
- **Voting Mechanism**: Combines predictions from 6 different models
- **Confidence Score**: Provides percentage confidence based on model agreement
- **Preprocessing Quality**: Advanced text preprocessing for better feature extraction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Data Sources

The training data is sourced from:
- Positive reviews: https://pythonprogramming.net/static/downloads/short_reviews/positive.txt
- Negative reviews: https://pythonprogramming.net/static/downloads/short_reviews/negative.txt

## 📧 Contact

For questions or suggestions, please open an issue on GitHub or reach out through:

- **GitHub**: [GregReynaldi](https://github.com/GregReynaldi)
- **Instagram**: [@gregoriusreyy](https://instagram.com/gregoriusreyy)
- **WeChat**: @gregoriusreyy

---